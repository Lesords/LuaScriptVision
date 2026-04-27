-- YOLO11 Classification Script
local Model = {}

Model.config = {
    input_size = {224, 224},
    topk = 5,
}

Model.preprocess_config = {
    type = "resize_center_crop",
    input_size = {224, 224}
}

-- ==========================================================
-- Pre-processing
-- ==========================================================
function Model.preprocess(img)
    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(Model.config.input_size)

    local scale = math.max(target_w / w, target_h / h)
    local new_w = math.floor(w * scale)
    local new_h = math.floor(h * scale)

    img:resize(new_w, new_h)

    if new_w ~= target_w or new_h ~= target_h then
        img:resize(target_w, target_h)
    end

    local input_tensor = img:to_tensor(1.0 / 255.0, {0, 0, 0}, {1, 1, 1})
    return input_tensor, {ori_w = w, ori_h = h}
end

-- ==========================================================
-- Post-processing
-- ==========================================================
function Model.postprocess(outputs, meta)
    local output_tensor = nil
    for k, v in pairs(outputs) do
        output_tensor = v
        break
    end

    if not output_tensor then
        error("No output tensor found")
    end

    local shape = output_tensor:shape()
    -- Model output is [1, N, 1, 1], reshape to [1, N] for topk
    if #shape == 4 and shape[3] == 1 and shape[4] == 1 then
        output_tensor = output_tensor:reshape({shape[1], shape[2]})
    end

    local topk_results = output_tensor:topk(Model.config.topk)

    local classes = {}
    local labels = {}
    for i = 1, #topk_results do
        local item = topk_results[i]
        local score_pct = math.floor(item.confidence * 100 + 0.5)
        table.insert(classes, {score_pct, item.class_id})
        -- Use node config labels (from Node-RED), fallback to "C<class_id>"
        local label = (meta.classes and meta.classes[item.class_id + 1])
                     or ("C" .. item.class_id)
        table.insert(labels, label)
    end

    return {classes = classes, labels = labels}
end

return Model
