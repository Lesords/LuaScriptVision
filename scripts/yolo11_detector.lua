-- Import our defined C++ modules (this line can be omitted if C++ is globally registered)
-- local cv = require "lua_cv"
-- local nn = require "lua_nn"
-- local utils = require "lua_utils"
local preprocess_lib = require("lib.preprocess")
local coco_labels = require("lib.coco")

local Model = {}

-- ==========================================================
-- 1. Model Configuration (Config)
-- ==========================================================
Model.config = {
    input_size = {640, 640},  -- [H, W]
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    -- COCO class labels
    labels = coco_labels
}

-- C++ Preprocess Configuration (使用C++预处理函数)
Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,
    fill_value = 114
}

-- ==========================================================
-- 2. Pre-processing (Lua fallback implementation)
-- Input: lua_cv.Image object
-- Output: lua_nn.Tensor object, meta information table
-- ==========================================================
function Model.preprocess(img)
    -- 使用公共库的letterbox函数
    return preprocess_lib.letterbox(img, Model.config.input_size, Model.config.stride)
end

-- ==========================================================
-- 3. Post-processing
-- Input: Inference result Map, meta information
-- Output: Final detection box list
-- ==========================================================
function Model.postprocess(outputs, meta)
    -- Get output Tensor
    local output_tensor = outputs["output0"] -- Assume model output node name is output0

    -- STEP 1: Quick filtering (Critical Optimization)
    -- Call C++ implemented filter to avoid Lua looping 25200 times
    -- Return format: { {x,y,w,h,score,class_id}, ... } in original model coordinates
    -- Note: C++ implementation now handles both YOLOv5 [1, N, 85] and YOLOv8/11 [1, 84, N] formats automatically
    local raw_boxes = output_tensor:filter_yolo(Model.config.conf_thres)

    local proposals = {}

    -- STEP 2: Coordinate restoration (only process the small number of filtered boxes)
    for _, box in ipairs(raw_boxes) do
        -- Reverse Letterbox calculation
        -- Box structure defined by C++: box.x, box.y, box.w, box.h, box.score, box.cls
        
        local x = (box.x - meta.pad_x) / meta.scale
        local y = (box.y - meta.pad_y) / meta.scale
        local w = box.w / meta.scale
        local h = box.h / meta.scale

        -- Boundary clipping (optional, prevent boxes from exceeding original image)
        x = math.max(0, x)
        y = math.max(0, y)
        w = math.min(w, meta.ori_w - x)
        h = math.min(h, meta.ori_h - y)

        table.insert(proposals, {
            x = x, y = y, w = w, h = h,
            score = box.score,
            label = Model.config.labels[box.cls + 1] -- Lua index starts from 1
        })
    end

    -- STEP 3: NMS (call lua_utils)
    local final_results = lua_utils.nms(proposals, Model.config.iou_thres)

    return final_results
end

return Model
