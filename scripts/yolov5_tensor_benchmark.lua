-- YOLOv5 Benchmark Script with Detailed Timing
-- 基于 yolov5_tensor_detector.lua，添加精确计时统计

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
    cvtcolor_times = {},
    to_blob_times = {},

    -- Postprocess
    slice_times = {},
    contiguous_times = {},
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
    local new_h, new_w = math.floor(h * r + 0.5), math.floor(w * r + 0.5)
    local pad_h, pad_w = target_h - new_h, target_w - new_w

    local top = math.floor(pad_h / 2)
    local bottom = pad_h - top
    local left = math.floor(pad_w / 2)
    local right = pad_w - left

    -- Timing: letterbox (resize + pad)
    local t1 = os.clock()
    img:resize(new_w, new_h)
    img:pad(top, bottom, left, right, 114)
    local t2 = os.clock()
    table.insert(Model.timings.letterbox_times, (t2 - t1) * 1000)

    -- Timing: to_tensor (combines cvtColor + normalization + blob creation)
    local t3 = os.clock()
    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})
    local t4 = os.clock()
    table.insert(Model.timings.cvtcolor_times, (t4 - t3) * 1000)  -- Reuse cvtcolor timing for to_tensor
    table.insert(Model.timings.to_blob_times, 0)  -- No separate to_blob in YOLOv5

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
    local output = outputs["output0"]:squeeze(0)  -- [25200, 85]

    -- Timing: extract columns
    local t1 = os.clock()
    local boxes_x = output:get_column(0)
    local boxes_y = output:get_column(1)
    local boxes_w = output:get_column(2)
    local boxes_h = output:get_column(3)
    local objectness = output:get_column(4)
    local class_scores = output:slice_columns(5, 85)  -- [25200, 80]
    local t2 = os.clock()
    table.insert(Model.timings.slice_times, (t2 - t1) * 1000)

    -- Timing: max_with_argmax (fused)
    local t3 = os.clock()
    local result = class_scores:max_with_argmax(1)
    local max_class_scores = result.values  -- [25200]
    local class_ids = result.indices  -- table[25200]
    local t4 = os.clock()
    table.insert(Model.timings.maxarg_times, (t4 - t3) * 1000)

    -- Note: contiguous timing is not applicable for YOLOv5 column-based approach
    table.insert(Model.timings.contiguous_times, 0)

    -- Timing: filtering + proposal building (combined in Lua loop for YOLOv5)
    local t5 = os.clock()
    local proposals = {}
    local num_boxes = 25200

    for i = 0, num_boxes - 1 do
        local obj = objectness:get(i)
        if obj >= Model.config.conf_thres then
            local max_cls_score = max_class_scores:get(i)
            local final_score = obj * max_cls_score

            if final_score >= Model.config.conf_thres then
                local cx = boxes_x:get(i)
                local cy = boxes_y:get(i)
                local w = boxes_w:get(i)
                local h = boxes_h:get(i)
                local cls_id = class_ids[i + 1]

                -- Convert center to top-left and scale back to original coordinates (使用公共库函数)
                local x = cx - w / 2
                local y = cy - h / 2
                x, y = preprocess_lib.scale_coords(x, y, meta)
                local scaled_w = preprocess_lib.scale_size(w, meta)
                local scaled_h = preprocess_lib.scale_size(h, meta)

                table.insert(proposals, {
                    x = x,
                    y = y,
                    w = scaled_w,
                    h = scaled_h,
                    score = final_score,
                    class_id = cls_id,
                    label = Model.config.labels[cls_id + 1] or "unknown"
                })
            end
        end
    end
    local t6 = os.clock()
    local loop_time = (t6 - t5) * 1000

    -- Split loop time into where/extract/loop for compatibility
    table.insert(Model.timings.where_times, loop_time * 0.3)
    table.insert(Model.timings.extract_times, loop_time * 0.2)
    table.insert(Model.timings.loop_times, loop_time * 0.5)

    if #proposals == 0 then
        table.insert(Model.timings.nms_times, 0)
        return {}
    end

    -- Timing: NMS
    local t7 = os.clock()
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    local t8 = os.clock()
    table.insert(Model.timings.nms_times, (t8 - t7) * 1000)

    -- Increment iteration counter and print summary if needed
    Model.iteration_count = Model.iteration_count + 1
    if Model.iteration_count % Model.print_every_n == 0 then
        Model.print_timing_summary()
    end

    return final_boxes
end

-- Print detailed timing summary
function Model.print_timing_summary()
    local config = {
        count_key = "slice_times",
        preprocess_items = {
            {"letterbox (resize + pad)", "letterbox_times"},
            {"to_tensor (cvt+norm+blob)", "cvtcolor_times"}
        },
        postprocess_items = {
            {"get_column + slice_columns", "slice_times"},
            {"max_with_argmax (fused)", "maxarg_times"},
            {"filter + build proposals (Lua)", {"where_times", "extract_times", "loop_times"}},
            {"NMS", "nms_times"}
        },
        notes = {
            "Compare with cpp_infer baseline (pure C++ YOLOv5)",
            "to_tensor combines cvtColor + normalization + blob creation"
        }
    }
    benchmark.print_timing_summary("YOLOv5", Model.timings, config)
end

return Model
