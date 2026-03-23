#pragma once

#include "shared_frame.h"

#include <nlohmann/json.hpp>

namespace node {

// PipelineContext carries frame data between nodes
// - Used by all DataNode types (CameraNode, ModelNode, etc.)
// - Move-only semantics to prevent double-release
// - Destructor automatically releases frame references
// - stream_frame is optional (nullptr when not populated)
struct PipelineContext {
    SharedFrame* frame;               // INFER channel frame (scaled, e.g. 640x640 RGB)
    SharedFrame* stream_frame;        // STREAM channel frame (full-res, e.g. 1920x1080 NV21), nullable
    nlohmann::json upstream_result;   // Result from upstream node (empty = from CameraNode)
    uint64_t frame_id;                // Frame sequence number

    // Destructor releases both frame references
    ~PipelineContext() {
        if (frame) {
            frame->release();
            frame = nullptr;
        }
        if (stream_frame) {
            stream_frame->release();
            stream_frame = nullptr;
        }
    }

    // Move-only: prevent copying to avoid double-release
    PipelineContext(const PipelineContext&) = delete;
    PipelineContext& operator=(const PipelineContext&) = delete;

    PipelineContext(PipelineContext&& other) noexcept
        : frame(other.frame),
          stream_frame(other.stream_frame),
          upstream_result(std::move(other.upstream_result)),
          frame_id(other.frame_id) {
        other.frame = nullptr;
        other.stream_frame = nullptr;
    }

    PipelineContext& operator=(PipelineContext&& other) noexcept {
        if (this != &other) {
            if (frame) frame->release();
            if (stream_frame) stream_frame->release();
            frame = other.frame;
            stream_frame = other.stream_frame;
            upstream_result = std::move(other.upstream_result);
            frame_id = other.frame_id;
            other.frame = nullptr;
            other.stream_frame = nullptr;
        }
        return *this;
    }

    // Constructor (without stream frame — backward compatible)
    // Note: Caller must call frame->ref() BEFORE creating PipelineContext
    PipelineContext(SharedFrame* f, nlohmann::json result, uint64_t id)
        : frame(f), stream_frame(nullptr), upstream_result(std::move(result)), frame_id(id) {}

    // Constructor with stream frame
    // Note: Caller must call ref() on BOTH frames before creating PipelineContext
    PipelineContext(SharedFrame* f, SharedFrame* sf, nlohmann::json result, uint64_t id)
        : frame(f), stream_frame(sf), upstream_result(std::move(result)), frame_id(id) {}

    // Check if this context came directly from CameraNode (no upstream result)
    bool is_from_camera() const {
        return upstream_result.empty() || upstream_result.is_null();
    }

    // Check if a full-resolution stream frame is available
    bool has_stream_frame() const { return stream_frame != nullptr; }
};

} // namespace node
