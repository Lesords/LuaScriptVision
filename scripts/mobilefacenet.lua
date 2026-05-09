-- MobileFaceNet 128D Face Embedding
-- Pipeline: SCRFD (FULL_FRAME) → MobileFaceNet (CROPPED_ROI)
-- Input: 112×112 RGB face crop from SCRFD detection
-- Output: 128D L2-normalized embedding per face

local Model = {}

Model.config = {
    min_face_size = 60,
    max_yaw_ratio = 0.5,  -- skip side faces: |d_left - d_right| / max(d_left, d_right)
}

Model.preprocess_config = {
    -- CPU inference path: face_align uses keypoints for affine warp → float tensor → quantize → TPU Forward.
    -- normalize must be false to avoid double normalization:
    -- CPU sends raw pixel values (0-255), TPU applies model's internal
    -- normalization (mean/scale embedded in cvimodel), matching VB-path behavior.
    type = "face_align",
    input_size = {112, 112},
    format = "hwc",
    normalize = false,
    save_crop = false,
    save_path = "/tmp",
}

function Model.select_rois(upstream)
    local rois = {}
    local boxes = upstream.boxes or {}
    print(string.format("[MobileFaceNet] select_rois: received %d boxes from upstream", #boxes))
    for _, box in ipairs(boxes) do
        if box.w >= Model.config.min_face_size and box.h >= Model.config.min_face_size then
            -- Filter side faces: check eye-to-nose symmetry
            local kps = box.keypoints
            if kps and #kps >= 3 then
                local dx_l = kps[1].x - kps[3].x
                local dy_l = kps[1].y - kps[3].y
                local dx_r = kps[2].x - kps[3].x
                local dy_r = kps[2].y - kps[3].y
                local d_left = math.sqrt(dx_l * dx_l + dy_l * dy_l)
                local d_right = math.sqrt(dx_r * dx_r + dy_r * dy_r)
                local d_max = math.max(d_left, d_right)
                if d_max > 0 then
                    local yaw_ratio = math.abs(d_left - d_right) / d_max
                    if yaw_ratio > Model.config.max_yaw_ratio then
                        print(string.format("[MobileFaceNet] skip side face: yaw_ratio=%.2f", yaw_ratio))
                        goto continue
                    end
                end
            end
            table.insert(rois, box)
        end
        ::continue::
    end
    print(string.format("[MobileFaceNet] select_rois: %d ROIs after filter", #rois))
    return rois
end

function Model.postprocess(outputs, meta)
    local emb_tensor = nil
    for _, v in pairs(outputs) do
        emb_tensor = v
        break
    end
    if not emb_tensor then
        print("[MobileFaceNet] postprocess: no output tensor")
        return {}
    end

    local shape = emb_tensor:shape()
    local total = 1
    for _, d in ipairs(shape) do total = total * d end
    print(string.format("[MobileFaceNet] postprocess: shape=%s total=%d",
          table.concat(shape, "x"), total))

    -- Reshape to [total, 1] for at2d binding
    emb_tensor = emb_tensor:reshape({total, 1})

    local sum_sq = 0.0
    for i = 0, total - 1 do
        local v = emb_tensor:at(i, 0)
        sum_sq = sum_sq + v * v
    end
    local norm = math.sqrt(sum_sq)
    if norm < 1e-10 then norm = 1.0 end
    print(string.format("[MobileFaceNet] postprocess: L2 norm=%.4f first3=[%.4f,%.4f,%.4f]",
          norm, emb_tensor:at(0, 0) / norm, emb_tensor:at(1, 0) / norm, emb_tensor:at(2, 0) / norm))

    local embedding = {}
    for i = 0, total - 1 do
        table.insert(embedding, emb_tensor:at(i, 0) / norm)
    end

    local roi = meta.roi or {}
    local source = meta.source or {}

    local result = {
        roi = roi,
        embedding = embedding,
    }
    result.score = source.score
    result.label = source.label or "face"
    result.keypoints = source.keypoints

    return result
end

return Model
