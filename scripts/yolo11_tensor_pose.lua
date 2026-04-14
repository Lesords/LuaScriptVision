-- YOLO11 Pose Estimation Script (使用新Tensor API)
-- 展示如何用通用Tensor操作处理姿态估计
local utils = lua_utils
local nn = lua_nn
local preprocess_lib = require("scripts.lib.preprocess")

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    num_keypoints = 17,  -- COCO格式17个关键点
    keypoint_names = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    },
    skeleton = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
        {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
        {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3},
        {2, 4}, {3, 5}, {4, 6}, {5, 7}
    }
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

-- 使用向量化Tensor操作实现姿态估计后处理（方案3 - 极致性能）
function Model.postprocess(outputs, meta)
    local t0 = os.clock()

    local output = outputs["output0"]

    if not output then
        error("Missing output0")
    end

    -- YOLO11-Pose格式: [1, 56, 8400] 或 [56, 8400]
    -- 4个box坐标 + 1个person类别分数 + 51个关键点数据(17个点*3: x,y,conf)
    local num_kpt = Model.config.num_keypoints
    local shape = output:shape()

    -- Normalize to 2D [channels, anchors] format
    -- Handles: [1, 56, 8400], [56, 8400], [1, 56, 8400, 1], [8400, 56]
    local channels, anchors

    if #shape == 4 then
        -- [1, C, N, 1] → squeeze batch & trailing dims
        output = output:squeeze(0):squeeze(-1):contiguous()  -- [C, N]
        channels, anchors = shape[2], shape[3]
    elseif #shape == 3 then
        if shape[2] < shape[3] then
            output = output:squeeze(0)  -- [C, N]
            channels, anchors = shape[2], shape[3]
        else
            output = output:squeeze(0):transpose(0, 1):contiguous()  -- [C, N]
            channels, anchors = shape[3], shape[2]
        end
    elseif #shape == 2 then
        if shape[1] < shape[2] then
            channels, anchors = shape[1], shape[2]
        else
            output = output:transpose(0, 1):contiguous()
            channels, anchors = shape[2], shape[1]
        end
    else
        error(string.format("Unexpected output shape: [%s]", table.concat(shape, ", ")))
    end

    -- 1. 分离boxes, scores, keypoints from [channels, anchors]
    local boxes = output:slice(0, 0, 4, 1):contiguous()           -- [4, anchors]
    local person_scores = output:slice(0, 4, 5, 1):reshape({anchors}):contiguous()  -- [anchors]
    local kpt_channels = num_kpt * 3  -- 51
    local keypoints = output:slice(0, 5, 5 + kpt_channels, 1):contiguous()  -- [51, anchors]

    local t1 = os.clock()

    -- 2. 向量化过滤：找出满足条件的索引
    local valid_indices = person_scores:where_indices(Model.config.conf_thres, "ge")

    print(string.format("过滤后候选框: %d", #valid_indices))

    if #valid_indices == 0 then
        return {}
    end

    -- 3. 直接遍历有效索引，使用 at() 高效访问元素（避免 to_table 转换）
    local proposals = {}

    for i = 1, #valid_indices do
        local col = valid_indices[i]  -- 0-based column index

        -- 直接从原始 boxes tensor 提取坐标 [4, 8400]
        local cx = boxes:at(0, col)
        local cy = boxes:at(1, col)
        local w = boxes:at(2, col)
        local h = boxes:at(3, col)
        local conf = person_scores:get(col)

        -- 将中心点坐标转换为左上角坐标
        local x = cx - w / 2.0
        local y = cy - h / 2.0

        -- 提取17个关键点(每个3个值: x, y, conf) 直接从 keypoints tensor [51, 8400]
        local kpts = {}
        for j = 1, num_kpt do
            local row_base = (j - 1) * 3
            local kpt_x = keypoints:at(row_base, col)
            local kpt_y = keypoints:at(row_base + 1, col)
            local kpt_c = keypoints:at(row_base + 2, col)

            kpts[j] = {
                x = kpt_x,
                y = kpt_y,
                v = kpt_c,
                conf = kpt_c,
                name = Model.config.keypoint_names[j]
            }
        end

        table.insert(proposals, {
            x = x,
            y = y,
            w = w,
            h = h,
            score = conf,
            class_id = 0,  -- person类别
            label = "person",
            keypoints = kpts
        })
    end
    
    -- 3. 坐标还原(包括关键点，使用公共库函数)
    for _, box in ipairs(proposals) do
        box.x, box.y = preprocess_lib.scale_coords(box.x, box.y, meta)
        box.w = preprocess_lib.scale_size_w(box.w, meta)
        box.h = preprocess_lib.scale_size_h(box.h, meta)

        -- 还原关键点坐标
        for _, kpt in ipairs(box.keypoints) do
            kpt.x, kpt.y = preprocess_lib.scale_coords(kpt.x, kpt.y, meta)
        end

        -- 边界裁剪
        preprocess_lib.clamp_box(box, meta)
    end
    
    -- 4. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)

    local t2 = os.clock()

    -- Debug: print first box + first 3 keypoints to verify coordinates
    if #final_boxes > 0 then
        local b = final_boxes[1]
        print(string.format("[Pose] tensor:%.1fms total:%.1fms box: %.0f,%.0f,%.0f,%.0f res:%dx%d",
              (t1 - t0) * 1000, (t2 - t0) * 1000, b.x, b.y, b.w, b.h,
              meta.ori_w or 0, meta.ori_h or 0))
        local kpts = b.keypoints
        if kpts and #kpts >= 3 then
            print(string.format("  kpt[1]=%.1f,%.1f kpt[2]=%.1f,%.1f kpt[6]=%.1f,%.1f",
                  kpts[1].x, kpts[1].y, kpts[2].x, kpts[2].y,
                  kpts[6].x, kpts[6].y))
        end
    else
        print(string.format("[Pose] tensor:%.1fms total:%.1fms boxes:0",
              (t1 - t0) * 1000, (t2 - t0) * 1000))
    end

    return final_boxes
end

return Model
