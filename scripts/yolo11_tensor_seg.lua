-- YOLO11 Instance Segmentation Script (使用新Tensor API)
-- 展示如何用通用Tensor操作处理分割任务
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
-- Note: C++ preprocess automatically adds input_w/input_h fields
Model.preprocess_config = {
    type = "letterbox",
    input_size = {640, 640},
    stride = 32,
    fill_value = 114
}

-- Lua fallback implementation (仅在C++预处理不可用时使用)
-- function Model.preprocess(img)
--     local input_tensor, meta = preprocess_lib.letterbox(img, Model.config.input_size, Model.config.stride)
--     meta.input_w = img.width
--     meta.input_h = img.height
--     return input_tensor, meta
-- end

-- 使用向量化Tensor操作实现分割后处理（方案3 - 极致性能）
function Model.postprocess(outputs, meta)
    local output0 = outputs["output0"]  -- [1, 116, 8400]: boxes + 32 mask coeffs
    local output1 = outputs["output1"]  -- [1, 32, 160, 160]: proto masks

    if not output0 or not output1 then
        error("Missing outputs")
    end

    -- YOLO11-Seg格式: output0=[1, 116, 8400], output1=[1, 32, 160, 160]
    local num_mask_coeffs = 32
    local num_classes = 80

    -- 1. 分离boxes, scores, mask_coeffs
    -- 关键优化：立即调用contiguous()确保后续操作高效
    local boxes = output0:slice(1, 0, 4, 1):squeeze(0):contiguous()  -- [4, 8400]
    local scores = output0:slice(1, 4, 4 + num_classes, 1):squeeze(0):contiguous()  -- [80, 8400]
    local mask_coeffs = output0:slice(1, 4 + num_classes, 116, 1):squeeze(0):contiguous()  -- [32, 8400]

    -- 2. 找到最大分数和类别 (融合操作，单次遍历)
    local result = scores:max_with_argmax(0)  -- {values=Tensor[8400], indices=table[8400]}
    local max_scores = result.values
    local class_ids = result.indices

    -- 3. 向量化过滤：找出满足条件的索引
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")

    print(string.format("过滤后候选框: %d", #valid_indices))

    if #valid_indices == 0 then
        return {}
    end

    -- 4. 直接遍历有效索引，使用 at() 高效访问元素（避免 to_table 转换）
    local proposals = {}

    for i = 1, #valid_indices do
        local col = valid_indices[i]  -- 0-based column index

        -- 直接从原始 boxes tensor 提取坐标 [4, 8400]
        local cx = boxes:at(0, col)
        local cy = boxes:at(1, col)
        local w = boxes:at(2, col)
        local h = boxes:at(3, col)
        local cls_id = class_ids[col + 1]  -- Lua table 索引从1开始
        local conf = max_scores:get(col)

        -- 将中心点坐标转换为左上角坐标
        local x = cx - w / 2.0
        local y = cy - h / 2.0

        -- 提取32个mask系数 (process_mask需要table)
        local coeffs = {}
        for j = 0, num_mask_coeffs - 1 do
            coeffs[j + 1] = mask_coeffs:at(j, col)
        end

        table.insert(proposals, {
            x = x,
            y = y,
            w = w,
            h = h,
            score = conf,
            class_id = cls_id,
            label = Model.config.labels[cls_id + 1] or "unknown",
            mask_coeffs = coeffs
        })
    end
    
    -- 4. 坐标还原（使用公共库函数）
    for _, box in ipairs(proposals) do
        box.x, box.y = preprocess_lib.scale_coords(box.x, box.y, meta)
        box.w = preprocess_lib.scale_size_w(box.w, meta)
        box.h = preprocess_lib.scale_size_h(box.h, meta)
        preprocess_lib.clamp_box(box, meta)
    end
    
    -- 5. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    print(string.format("NMS后最终框: %d", #final_boxes))
    
    -- 6. 生成mask (使用legacy方法，因为mask生成涉及复杂的矩阵乘法)
    -- TODO: 后续可以用tensor操作实现 mask = sigmoid(coeffs @ proto_masks)
    for i = 1, #final_boxes do
        local box = final_boxes[i]
        -- 暂时使用旧的process_mask方法
        local mask_tensor = output1:process_mask(
            box.mask_coeffs, box, 
            meta.ori_w, meta.ori_h, 
            meta.input_w, meta.input_h, 
            meta.pad_x, meta.pad_y
        )
        box.mask = mask_tensor
        box.mask_coeffs = nil
    end
    
    return final_boxes
end

return Model
