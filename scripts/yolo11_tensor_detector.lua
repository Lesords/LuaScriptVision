-- YOLO11 Object Detection Script (Tensor API)
-- Supports both fused single-output and multi-head 6-output models
-- Optimized: vectorized DFL decode using pure C++ tensor operations
local utils = lua_utils
local nn = lua_nn
local preprocess_lib = require("scripts.lib.preprocess")
local coco_labels = require("scripts.lib.coco")

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    labels = coco_labels,
    reg_max = 16,
    strides = {8, 16, 32}
}

-- C++ Preprocess Configuration
Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,
    fill_value = 114
}

-- DFL decode using fused weighted_sum (RVV optimized on T-Head platforms)
-- Input: bbox_selected [64, n_valid] - selected valid anchors
-- Output: tensor [4, n_valid] with decoded distances
local function dfl_decode_vectorized(bbox_selected, reg_max, n_valid)
    -- 1. Reshape to [4, 16, n_valid]
    local reshaped = bbox_selected:reshape({4, reg_max, n_valid})

    -- 2. Create weights [0, 1, 2, ..., 15]
    local weights = nn.Tensor.arange(0, reg_max, 1)  -- [16]

    -- 3. Fused softmax + weighted sum (single C++ call, RVV optimized)
    local result = reshaped:weighted_sum(1, weights)  -- [4, n_valid]

    return result
end

-- Process multi-head YOLO output (6 outputs: 3 bbox + 3 cls)
-- Optimizations:
--   1. Vectorized DFL decode in pure C++
--   2. Single-pass sigmoid_max_with_argmax (RVV optimized, 13.87ms at memory bandwidth limit)
local function process_multi_head(outputs, meta)
    local reg_max = Model.config.reg_max
    local strides = Model.config.strides
    local conf_thres = Model.config.conf_thres

    local all_proposals = {}

    local t_total = os.clock()
    local time_reshape = 0
    local time_sigmoid_max = 0
    local time_filter = 0
    local time_dfl = 0
    local time_dist2bbox = 0

    -- Process each scale (3 scales)
    for scale = 0, 2 do
        local bbox_key = "output" .. (scale * 2)
        local cls_key = "output" .. (scale * 2 + 1)

        local bbox_output = outputs[bbox_key]
        local cls_output = outputs[cls_key]

        if not bbox_output or not cls_output then
            print(string.format("Warning: Missing output at scale %d", scale))
            goto continue
        end

        local bbox_shape = bbox_output:shape()
        local cls_shape = cls_output:shape()
        local stride = strides[scale + 1]

        -- Get grid dimensions from shape [1, channels, H, W]
        local grid_h = bbox_shape[3]
        local grid_w = bbox_shape[4]
        local n_anchors = grid_h * grid_w

        -- Reshape to [channels, n_anchors]
        local t0 = os.clock()
        local bbox_flat = bbox_output:reshape({bbox_shape[2], n_anchors}):contiguous()
        local cls_flat = cls_output:reshape({cls_shape[2], n_anchors}):contiguous()
        time_reshape = time_reshape + (os.clock() - t0)

        -- Single-pass sigmoid_max_with_argmax (RVV optimized)
        t0 = os.clock()
        local result = cls_flat:sigmoid_max_with_argmax(0)
        local max_scores = result.values     -- [n_anchors], sigmoid values
        local class_ids = result.indices     -- table[n_anchors], class indices
        local t_sigmoid_max_scale = (os.clock() - t0) * 1000
        time_sigmoid_max = time_sigmoid_max + (os.clock() - t0)

        -- Filter by confidence threshold
        t0 = os.clock()
        local valid_indices = max_scores:where_indices(conf_thres, "ge")
        time_filter = time_filter + (os.clock() - t0)

        if #valid_indices == 0 then
            goto continue
        end

        local n_valid = #valid_indices

        -- Extract valid anchors for DFL decode
        t0 = os.clock()
        local bbox_selected = bbox_flat:index_select(1, valid_indices)  -- [64, n_valid]

        -- VECTORIZED DFL DECODE: Pure C++ tensor operations
        local dist_decoded = dfl_decode_vectorized(bbox_selected, reg_max, n_valid)  -- [4, n_valid]

        -- Get filtered scores
        local filtered_scores = max_scores:index_select(0, valid_indices):to_table()
        time_dfl = time_dfl + (os.clock() - t0)

        -- Convert decoded distances to boxes
        t0 = os.clock()
        for i = 1, n_valid do
            -- valid_indices[i] is 0-based C++ index
            local anchor_idx = valid_indices[i]

            -- Get decoded distances
            local l = dist_decoded:at(0, i - 1)
            local t = dist_decoded:at(1, i - 1)
            local r = dist_decoded:at(2, i - 1)
            local b = dist_decoded:at(3, i - 1)

            -- Calculate anchor center
            local row = math.floor(anchor_idx / grid_w)
            local col = anchor_idx % grid_w
            local ax = (col + 0.5) * stride
            local ay = (row + 0.5) * stride

            -- dist2bbox
            local x1 = ax - l * stride
            local y1 = ay - t * stride
            local x2 = ax + r * stride
            local y2 = ay + b * stride

            local cx = (x1 + x2) * 0.5
            local cy = (y1 + y2) * 0.5
            local w = x2 - x1
            local h = y2 - y1

            -- Get class ID (class_ids is 1-based Lua table)
            local cls_id = class_ids[anchor_idx + 1]
            local conf = filtered_scores[i]

            table.insert(all_proposals, {
                x = cx - w / 2.0,
                y = cy - h / 2.0,
                w = w,
                h = h,
                score = conf,
                class_id = cls_id,
                label = Model.config.labels[cls_id + 1] or "unknown"
            })
        end
        time_dist2bbox = time_dist2bbox + (os.clock() - t0)

        ::continue::
    end

    return all_proposals
end

-- Process fused single-output YOLO (output0 with shape [1, 84, 8400])
local function process_fused(outputs, meta)
    local output = outputs["output0"]

    -- YOLO11 fused format: [1, 84, 8400]
    -- First 4 channels: cx, cy, w, h (already decoded)
    -- Next 80 channels: class scores (already sigmoid-ed)
    local boxes = output:slice(1, 0, 4, 1):squeeze(0):contiguous()
    local scores = output:slice(1, 4, 84, 1):squeeze(0):contiguous()

    -- Find max class score and id
    local result = scores:max_with_argmax(0)
    local max_scores = result.values
    local class_ids = result.indices

    -- Filter by confidence
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")

    if #valid_indices == 0 then
        return {}
    end

    -- Extract filtered boxes and scores
    local filtered_boxes = boxes:extract_columns(valid_indices)
    local filtered_scores = max_scores:index_select(0, valid_indices):to_table()

    local proposals = {}
    for i = 1, #valid_indices do
        local idx = valid_indices[i]
        local box_data = filtered_boxes[i]

        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]
        local cls_id = class_ids[idx + 1]
        local conf = filtered_scores[i]

        table.insert(proposals, {
            x = cx - w / 2.0,
            y = cy - h / 2.0,
            w = w,
            h = h,
            score = conf,
            class_id = cls_id,
            label = Model.config.labels[cls_id + 1] or "unknown"
        })
    end

    return proposals
end

-- Main postprocess function
function Model.postprocess(outputs, meta)
    local output_count = meta.output_count or 1

    local proposals
    if output_count == 6 then
        -- Multi-head YOLO (6 outputs: 3 bbox + 3 cls)
        proposals = process_multi_head(outputs, meta)
    else
        -- Fused single-output YOLO
        proposals = process_fused(outputs, meta)
    end

    -- Scale coordinates back to original image
    for _, box in ipairs(proposals) do
        box.x, box.y = preprocess_lib.scale_coords(box.x, box.y, meta)
        box.w = preprocess_lib.scale_size_w(box.w, meta)
        box.h = preprocess_lib.scale_size_h(box.h, meta)
        preprocess_lib.clamp_box(box, meta)
    end

    -- Apply NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)

    print(string.format("NMS final boxes: %d", #final_boxes))

    return final_boxes
end

return Model
