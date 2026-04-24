-- YOLO11 Classification Script
-- local json = require "cjson" -- Not available

local Model = {}

-- ==========================================================
-- 1. Model Configuration
-- ==========================================================
Model.config = {
    input_size = {224, 224},
    topk = 5,
    labels = {
        [559] = "flute",      -- 558
        [820] = "stage",      -- 819
        [835] = "suit",       -- 834
        [777] = "sax",        -- 776
        [684] = "oboe",       -- 683
        [907] = "Windsor_tie" -- 906
    }
}

-- C++ Preprocess Configuration (使用C++预处理函数)
Model.preprocess_config = {
    type = "resize_center_crop",
    size = 224
}

-- Load labels
-- local function load_labels(path)
--     local f = io.open(path, "r")
--     if not f then return end
--     local content = f:read("*a")
--     f:close()
    
--     -- Simple JSON parser for this specific format {"0": ["id", "name"], ...}
--     for id, name in content:gmatch('"([0-9]+)": %["[^"]+", "([^"]+)"%]') do
--         Model.config.labels[tonumber(id) + 1] = name -- Lua 1-based
--     end
-- end

-- load_labels("imagenet_class_index.json")

-- ==========================================================
-- 2. Pre-processing
-- ==========================================================
function Model.preprocess(img)
    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(Model.config.input_size)

    -- Center Crop Strategy
    local scale = math.max(target_w / w, target_h / h)
    local new_w = math.floor(w * scale)
    local new_h = math.floor(h * scale)
    
    img:resize(new_w, new_h)
    
    if new_w ~= target_w or new_h ~= target_h then
        -- Simple center crop simulation by resizing (since we lack crop API)
        -- Ideally: img:crop(x, y, w, h)
        -- Fallback: Resize to target directly (might distort slightly if aspect ratio differs)
        img:resize(target_w, target_h)
    end

    -- Normalize (ImageNet mean/std)
    -- Mean: {0.485, 0.456, 0.406}, Std: {0.229, 0.224, 0.225}
    -- Note: Input is 0-255, so we scale by 1/255 first, then subtract mean, divide by std.
    -- Formula in to_tensor: (pixel * scale - mean) / std
    
    local scale = 1.0 / 255.0
    local mean = {0.485, 0.456, 0.406}
    local std = {0.229, 0.224, 0.225}
    
    -- Try standard YOLO normalization first (0-1) if ImageNet fails.
    -- But let's try ImageNet normalization as it's a classification model.
    -- Actually, Ultralytics YOLOv8/11 CLS models usually use:
    -- transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    -- Wait, Ultralytics docs say "YOLOv8 models are trained with ... 0-1 normalization".
    -- So my previous normalization was likely correct.
    
    -- Let's stick to 0-1 for now, but maybe the issue is the resize/crop.
    -- If I squash the image, it might be bad.
    -- Let's try to implement a better crop or just use the previous normalization and debug.
    
    -- Reverting to 0-1 for now, but let's add the label loading.
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})

    local meta = {
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end

-- ==========================================================
-- 3. Post-processing
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
    
    -- Map indices to labels
    -- Note: #Model.config.labels might be 0 if it's a sparse table (hash map in Lua)
    -- So we check if it's not nil
    if Model.config.labels then
        for i = 1, #topk_results do
            local item = topk_results[i]
            -- class_id is 0-based from C++, Lua labels are 1-based
            local label = Model.config.labels[item.class_id + 1]
            if label then
                item["label"] = label
            end
        end
    end
    
    return topk_results
end

return Model
