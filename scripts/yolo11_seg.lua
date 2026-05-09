-- YOLO11 Instance Segmentation Script
-- local utils = require "lua_utils"
-- local nn = require "lua_nn"
local utils = lua_utils
local nn = lua_nn
local preprocess_lib = require("lib.preprocess")
local coco_labels = require("lib.coco")

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

-- Lua fallback implementation (仅在C++预处理不可用时使用)
-- Note: C++ preprocess doesn't add input_w/input_h, but they match ori_w/ori_h
-- function Model.preprocess(img)
--     local input_tensor, meta = preprocess_lib.letterbox(img, Model.config.input_size, Model.config.stride)
--     meta.input_w = img.width
--     meta.input_h = img.height
--     return input_tensor, meta
-- end

function Model.postprocess(outputs, meta)
    -- outputs: output0 (boxes+masks_coeffs), output1 (proto_masks)
    local output0 = outputs["output0"]
    local output1 = outputs["output1"]
    
    if not output0 or not output1 then error("Missing outputs") end

    -- 1. Filter Boxes
    local raw_boxes = output0:filter_yolo_seg(Model.config.conf_thres)

    local proposals = {}

    for _, box in ipairs(raw_boxes) do
        -- Restore Box (使用公共库函数)
        local x, y = preprocess_lib.scale_coords(box.x, box.y, meta)
        local w = preprocess_lib.scale_size(box.w, meta)
        local h = preprocess_lib.scale_size(box.h, meta)
        
        table.insert(proposals, {
            x = x, y = y, w = w, h = h,
            score = box.score,
            class_id = box.class_id,
            label = Model.config.labels[box.class_id + 1] or "unknown",
            mask_coeffs = box.mask_coeffs
        })
    end

    -- 2. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    -- 3. Process Masks for final boxes
    for i = 1, #final_boxes do
        local box = final_boxes[i]
        -- Generate mask using C++ helper
        -- Note: We pass the actual input tensor size (which might be different from config due to stride alignment)
        local mask_tensor = output1:process_mask(box.mask_coeffs, box, meta.ori_w, meta.ori_h, meta.input_w, meta.input_h, meta.pad_x, meta.pad_y)
        box.mask = mask_tensor
        box.mask_coeffs = nil -- Clean up
    end
    
    return final_boxes
end

return Model