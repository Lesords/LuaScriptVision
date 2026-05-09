-- YOLO11 Pose Estimation Script
-- local utils = require "lua_utils" -- lua_utils is registered globally
local preprocess_lib = require("lib.preprocess")

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    labels = {"person"}
}

-- C++ Preprocess Configuration (使用C++预处理函数)
Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,
    fill_value = 114
}

-- Lua fallback implementation (仅在C++预处理不可用时使用)
-- function Model.preprocess(img)
--     return preprocess_lib.letterbox(img, Model.config.input_size, Model.config.stride)
-- end

function Model.postprocess(outputs, meta)
    local output_tensor = nil
    for k, v in pairs(outputs) do
        output_tensor = v
        break
    end
    
    if not output_tensor then error("No output tensor") end

    -- Use new C++ method for Pose
    local raw_boxes = output_tensor:filter_yolo_pose(Model.config.conf_thres)

    local proposals = {}

    for _, box in ipairs(raw_boxes) do
        -- Restore Box (使用公共库函数)
        local x, y = preprocess_lib.scale_coords(box.x, box.y, meta)
        local w = preprocess_lib.scale_size(box.w, meta)
        local h = preprocess_lib.scale_size(box.h, meta)
        
        -- Restore Keypoints (使用公共库函数)
        local kpts = box.keypoints
        for i = 1, #kpts do
            kpts[i].x, kpts[i].y = preprocess_lib.scale_coords(kpts[i].x, kpts[i].y, meta)
        end
        
        table.insert(proposals, {
            x = x, y = y, w = w, h = h,
            score = box.score,
            class_id = box.class_id,
            label = "person",
            keypoints = kpts
        })
    end

    -- NMS
    local final_boxes = lua_utils.nms(proposals, Model.config.iou_thres)
    
    return final_boxes
end

return Model