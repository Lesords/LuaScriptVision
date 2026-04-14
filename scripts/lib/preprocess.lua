-- Common Preprocessing Functions for YOLO Models

local M = {}

-- Letterbox预处理：缩放+padding
-- 保持宽高比的同时将图片缩放到目标大小
-- 参数:
--   img: Image对象
--   input_size: {height, width} 目标尺寸
--   stride: padding对齐步长 (默认32)
--   fill_value: padding填充值 (默认114)
-- 返回:
--   input_tensor: 预处理后的tensor
--   meta: 元数据 {scale, pad_x, pad_y, ori_w, ori_h}
function M.letterbox(img, input_size, stride, fill_value)
    stride = stride or 32
    fill_value = fill_value or 114

    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(input_size)

    -- 计算缩放比例（保持宽高比）
    local r = math.min(target_h / h, target_w / w)
    local new_w = math.floor(w * r)
    local new_h = math.floor(h * r)

    -- 缩放图片
    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h)
    end

    -- 计算padding（对齐到stride）
    local dw = target_w - new_w
    local dh = target_h - new_h

    dw = dw % stride
    dh = dh % stride

    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left

    -- 添加padding
    img:pad(top, bottom, left, right, fill_value)

    -- 转换为tensor
    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})

    -- 返回元数据用于后处理坐标转换
    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end

-- 坐标缩放和去padding
-- 将模型输出的坐标转换回原始图片坐标
-- 支持非均匀缩放 (scale_x != scale_y, 如camera VPSS resize)
function M.scale_coords(x, y, meta)
    local sx = meta.scale_x or meta.scale
    local sy = meta.scale_y or meta.scale
    local scaled_x = (x - (meta.pad_x or 0)) / sx
    local scaled_y = (y - (meta.pad_y or 0)) / sy
    return scaled_x, scaled_y
end

-- 宽度缩放（用于box的w）
function M.scale_size_w(size, meta)
    return size / (meta.scale_x or meta.scale)
end

-- 高度缩放（用于box的h）
function M.scale_size_h(size, meta)
    return size / (meta.scale_y or meta.scale)
end

-- 通用尺寸缩放（向后兼容，均匀缩放场景）
function M.scale_size(size, meta)
    return size / meta.scale
end

-- 边界裁剪：确保框和关键点不超出图片范围
function M.clamp_box(box, meta)
    local ori_w = meta.ori_w or meta.frame_width or 0
    local ori_h = meta.ori_h or meta.frame_height or 0
    if ori_w <= 0 or ori_h <= 0 then return end

    box.x = math.max(0, math.min(box.x, ori_w - 1))
    box.y = math.max(0, math.min(box.y, ori_h - 1))
    if box.x + box.w > ori_w then box.w = ori_w - box.x end
    if box.y + box.h > ori_h then box.h = ori_h - box.y end

    if box.keypoints then
        for _, kpt in ipairs(box.keypoints) do
            kpt.x = math.max(0, math.min(kpt.x, ori_w - 1))
            kpt.y = math.max(0, math.min(kpt.y, ori_h - 1))
        end
    end
end

return M
