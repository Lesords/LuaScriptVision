-- YOLO11 Benchmark Script with Detailed Timing
-- 基于yolo11_tensor_detector.lua，添加详细的计时统计

local utils = lua_utils
local nn = lua_nn
local preprocess_lib = require("lib.preprocess")
local coco_labels = require("lib.coco")
local benchmark = require("lib.benchmark")

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    labels = coco_labels  -- 使用公共库中的COCO labels
}

-- C++ Preprocess Configuration (使用C++预处理函数)
Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,
    fill_value = 114
}

-- Timing accumulators
Model.timings = {
    -- Preprocess
    letterbox_times = {},
    to_tensor_times = {},

    -- Postprocess
    slice_contiguous_times = {},
    maxarg_times = {},
    where_times = {},
    extract_times = {},
    loop_times = {},
    nms_times = {}
}

-- Iteration counter for benchmarking
Model.iteration_count = 0
Model.print_every_n = 1  -- Print timing summary after each iteration for benchmarking

-- Lua fallback implementation with detailed timing (仅用于预处理性能分析)
-- Note: C++ preprocess is faster but doesn't provide timing breakdown
-- Uncomment this function to benchmark preprocess stages separately
--[[
function Model.preprocess(img)
    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(Model.config.input_size)

    local r = math.min(target_h / h, target_w / w)
    local new_w = math.floor(w * r)
    local new_h = math.floor(h * r)

    local dw = target_w - new_w
    local dh = target_h - new_h

    dw = dw % Model.config.stride
    dh = dh % Model.config.stride

    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left

    -- Timing: letterbox (resize + pad are in-place operations)
    local t1 = os.clock()
    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h)
    end
    img:pad(top, bottom, left, right, 114)
    local t2 = os.clock()
    table.insert(Model.timings.letterbox_times, (t2 - t1) * 1000)

    -- Timing: to_tensor (combines cvtColor + normalization + blob creation)
    local t3 = os.clock()
    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})
    local t4 = os.clock()
    table.insert(Model.timings.to_tensor_times, (t4 - t3) * 1000)

    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end
--]]

function Model.postprocess(outputs, meta)
    local output = outputs["output0"]

    if not output then
        error("Missing output0")
    end

    -- YOLO11格式: [1, 84, 8400]
    local num_classes = 80

    -- Timing: slice + contiguous
    local t1 = os.clock()
    local boxes = output:slice(1, 0, 4, 1):squeeze(0):contiguous()  -- [4, 8400]
    local scores = output:slice(1, 4, 84, 1):squeeze(0):contiguous()  -- [80, 8400]
    local t2 = os.clock()
    table.insert(Model.timings.slice_contiguous_times, (t2 - t1) * 1000)

    -- Timing: max_with_argmax (fused)
    local t3 = os.clock()
    local result = scores:max_with_argmax(0)
    local max_scores = result.values
    local class_ids = result.indices
    local t4 = os.clock()
    table.insert(Model.timings.maxarg_times, (t4 - t3) * 1000)

    -- Timing: where_indices
    local t5 = os.clock()
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")
    local t6 = os.clock()
    table.insert(Model.timings.where_times, (t6 - t5) * 1000)

    if #valid_indices == 0 then
        table.insert(Model.timings.extract_times, 0)
        table.insert(Model.timings.loop_times, 0)
        table.insert(Model.timings.nms_times, 0)
        Model.iteration_count = Model.iteration_count + 1
        if Model.iteration_count % Model.print_every_n == 0 then
            Model.print_timing_summary()
        end
        return {}
    end

    -- Timing: extract_columns + index_select
    local t7 = os.clock()
    local filtered_boxes = boxes:extract_columns(valid_indices)
    local filtered_scores_tensor = max_scores:index_select(0, valid_indices)
    local filtered_scores = filtered_scores_tensor:to_table()
    local t8 = os.clock()
    table.insert(Model.timings.extract_times, (t8 - t7) * 1000)

    -- Timing: build proposals + coordinate transform
    local t9 = os.clock()
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

        local x = cx - w / 2.0
        local y = cy - h / 2.0

        table.insert(proposals, {
            x = x,
            y = y,
            w = w,
            h = h,
            score = conf,
            class_id = cls_id,
            label = Model.config.labels[cls_id + 1] or "unknown"
        })
    end

    -- Coordinate transform to original image (使用公共库函数)
    for _, box in ipairs(proposals) do
        box.x, box.y = preprocess_lib.scale_coords(box.x, box.y, meta)
        box.w = preprocess_lib.scale_size(box.w, meta)
        box.h = preprocess_lib.scale_size(box.h, meta)
    end
    local t10 = os.clock()
    table.insert(Model.timings.loop_times, (t10 - t9) * 1000)

    -- Timing: NMS
    local t11 = os.clock()
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    local t12 = os.clock()
    table.insert(Model.timings.nms_times, (t12 - t11) * 1000)

    -- Increment iteration counter and print summary if needed
    Model.iteration_count = Model.iteration_count + 1
    if Model.iteration_count % Model.print_every_n == 0 then
        Model.print_timing_summary()
    end

    return final_boxes
end

-- Print timing summary
function Model.print_timing_summary()
    local config = {
        count_key = "slice_contiguous_times",
        preprocess_items = {
            {"letterbox (resize + pad)", "letterbox_times"},
            {"to_tensor (cvt+norm+blob)", "to_tensor_times"}
        },
        postprocess_items = {
            {"slice + contiguous", "slice_contiguous_times"},
            {"max_with_argmax (fused)", "maxarg_times"},
            {"where_indices", "where_times"},
            {"extract_columns + index_select", "extract_times"},
            {"build proposals (Lua loop)", "loop_times"},
            {"NMS", "nms_times"}
        },
        notes = {
            "YOLO11 uses transpose-free slice operations for better performance"
        }
    }
    benchmark.print_timing_summary("YOLO11", Model.timings, config)
end

return Model
