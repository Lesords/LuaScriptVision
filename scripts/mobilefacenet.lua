-- MobileFaceNet 128D Face Embedding
-- Pipeline: SCRFD (FULL_FRAME) → MobileFaceNet (CROPPED_ROI)
-- Input: 112×112 RGB face crop from SCRFD detection
-- Output: 128D L2-normalized embedding per face

local Model = {}

Model.config = {
    min_face_size = 20,
}

Model.preprocess_config = {
    type = "resize",
    input_size = {112, 112},
}

function Model.select_rois(upstream)
    local rois = {}
    local boxes = upstream.boxes or {}
    print(string.format("[MobileFaceNet] select_rois: received %d boxes from upstream", #boxes))
    for _, box in ipairs(boxes) do
        if box.w >= Model.config.min_face_size and box.h >= Model.config.min_face_size then
            table.insert(rois, {x = box.x, y = box.y, w = box.w, h = box.h})
        end
    end
    print(string.format("[MobileFaceNet] select_rois: %d ROIs after min_size filter", #rois))
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

    -- Match ROI back to original SCRFD box for metadata
    local roi = meta.roi or {}
    local matched_box = nil
    local upstream_boxes = (meta.upstream and meta.upstream.boxes) or {}
    for _, box in ipairs(upstream_boxes) do
        if box.x == roi.x and box.y == roi.y and box.w == roi.w and box.h == roi.h then
            matched_box = box
            break
        end
    end

    local result = {
        roi = roi,
        embedding = embedding,
    }
    if matched_box then
        result.score = matched_box.score
        result.label = "face"
        result.keypoints = matched_box.keypoints
    end

    return result
end

return Model
