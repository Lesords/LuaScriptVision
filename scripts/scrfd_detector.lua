-- SCRFD Face Detection Script (with 5-point landmarks)
-- Supports SCRFD-500M-KPS model with 9 outputs:
--   3 score (sigmoided), 3 bbox (distance-based), 3 keypoints (5 points × 2 coords)
local utils = lua_utils
local preprocess_lib = require("lib.preprocess")

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.5,
    iou_thres  = 0.4,
    num_anchors = 2,
    strides = {8, 16, 32},
}

Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,
    fill_value = 0,
}

-- Classify output tensors by shape and sort by d0 descending (stride asc)
-- Deduplicates by (d0, d1): C++ layer adds two keys per tensor ("outputN" + model name).
-- Garbage INT8 dequant tensors are already filtered at C++ level.
local function classify_outputs(outputs)
    local scores = {}
    local bboxes = {}
    local kps_list = {}
    local seen = {}

    for name, tensor in pairs(outputs) do
        local shape = tensor:shape()
        if #shape >= 2 then
            local d0 = shape[1]
            local d1 = shape[2]
            local key = d0 .. "_" .. d1
            if seen[key] then goto skip end
            seen[key] = true
            if d1 == 1 then
                table.insert(scores, {data = tensor, d0 = d0})
            elseif d1 == 4 then
                table.insert(bboxes, {data = tensor, d0 = d0})
            elseif d1 == 10 then
                table.insert(kps_list, {data = tensor, d0 = d0})
            end
        end
        ::skip::
    end

    local function sort_desc(a, b) return a.d0 > b.d0 end
    table.sort(scores, sort_desc)
    table.sort(bboxes, sort_desc)
    table.sort(kps_list, sort_desc)

    return scores, bboxes, kps_list
end

function Model.postprocess(outputs, meta)
    local scores, bboxes, kps_list = classify_outputs(outputs)

    if #scores ~= 3 or #bboxes ~= 3 or #kps_list ~= 3 then
        print(string.format("SCRFD: expected 3+3+3 outputs, got %d+%d+%d",
              #scores, #bboxes, #kps_list))
        return {}
    end

    local strides = Model.config.strides
    local num_anchors = Model.config.num_anchors
    local all_proposals = {}

    for s = 1, 3 do
        local stride = strides[s]
        local score_tensor = scores[s].data
        local bbox_tensor = bboxes[s].data
        local kps_tensor = kps_list[s].data

        local n_rows = scores[s].d0
        local feat_w = Model.config.input_size[1] / stride
        local feat_h = Model.config.input_size[2] / stride

        -- Squeeze shape [N,1,1,1] → [N,1] (keep 2D for at2d binding)
        local score_flat = score_tensor:reshape({n_rows, 1})
        local bbox_flat = bbox_tensor:reshape({n_rows, 4})
        local kps_flat = kps_tensor:reshape({n_rows, 10})

        -- Score statistics
        local s_min, s_max = score_flat:at(0, 0), score_flat:at(0, 0)
        local s_sum, s_cnt = 0, 0
        local s_above_03, s_above_05 = 0, 0
        for i = 0, n_rows - 1 do
            local v = score_flat:at(i, 0)
            if v < s_min then s_min = v end
            if v > s_max then s_max = v end
            s_sum = s_sum + v
            s_cnt = s_cnt + 1
            if v >= 0.3 then s_above_03 = s_above_03 + 1 end
            if v >= 0.5 then s_above_05 = s_above_05 + 1 end
        end
        print(string.format("  stride=%d score: min=%.4f max=%.4f mean=%.4f >=0.3:%d >=0.5:%d",
              stride, s_min, s_max, s_sum/s_cnt, s_above_03, s_above_05))

        -- Filter by confidence threshold
        local valid_indices = score_flat:where_indices(Model.config.conf_thres, "ge")
        if #valid_indices == 0 then goto continue end

        for vi = 1, #valid_indices do
            local idx = valid_indices[vi]  -- 0-based
            local conf = score_flat:at(idx, 0)

            -- Anchor center
            local anchor = idx % num_anchors
            local spatial = math.floor(idx / num_anchors)
            local col = spatial % feat_w
            local row = math.floor(spatial / feat_w)
            local cx = col * stride + anchor * 0.5 * stride
            local cy = row * stride + anchor * 0.5 * stride

            -- Distance-based bbox decode
            local dl = bbox_flat:at(idx, 0) * stride
            local dt = bbox_flat:at(idx, 1) * stride
            local dr = bbox_flat:at(idx, 2) * stride
            local db = bbox_flat:at(idx, 3) * stride

            local x1 = cx - dl
            local y1 = cy - dt
            local x2 = cx + dr
            local y2 = cy + db

            local bw = x2 - x1
            local bh = y2 - y1

            -- 5-point keypoints
            local keypoints = {}
            for k = 0, 4 do
                local kx = cx + kps_flat:at(idx, k * 2) * stride
                local ky = cy + kps_flat:at(idx, k * 2 + 1) * stride
                table.insert(keypoints, {x = kx, y = ky})
            end

            table.insert(all_proposals, {
                x = x1,
                y = y1,
                w = bw,
                h = bh,
                score = conf,
                class_id = 0,
                label = "face",
                keypoints = keypoints,
            })
        end

        ::continue::
    end

    -- Scale coordinates back to original image
    for _, box in ipairs(all_proposals) do
        box.x, box.y = preprocess_lib.scale_coords(box.x, box.y, meta)
        box.w = preprocess_lib.scale_size_w(box.w, meta)
        box.h = preprocess_lib.scale_size_h(box.h, meta)
        if box.keypoints then
            local sx = meta.scale_x or meta.scale
            local sy = meta.scale_y or meta.scale
            for _, kpt in ipairs(box.keypoints) do
                kpt.x = (kpt.x - (meta.pad_x or 0)) / sx
                kpt.y = (kpt.y - (meta.pad_y or 0)) / sy
            end
        end
        preprocess_lib.clamp_box(box, meta)
    end

    local final_boxes = utils.nms(all_proposals, Model.config.iou_thres)
    print(string.format("SCRFD: %d faces after NMS", #final_boxes))

    return final_boxes
end

return Model
