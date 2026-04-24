#pragma once

#include <cstddef>

namespace node {

// SG2002 hardware resource limits
constexpr int VB_POOL4_TOTAL = 4;              // Pool 4 (640x640 RGB) infer-only, 4 blocks (depth=2+hold=1+margin=1)
constexpr int VB_POOL3_TOTAL = 1;              // Pool 3 (640x640 RGB) VPSS-preprocess, 1 block (serialized VPSS)
constexpr int VB_BUFFER_RESERVE = 1;           // Reserved buffer for processing

// Model topology limits (based on VB constraints)
constexpr int MAX_PARALLEL_MODELS = 3;         // Parallel mode: max 3 models
constexpr int MAX_SERIAL_MODELS = 3;           // Serial mode: max 3 models

// Memory limits
constexpr size_t ION_TOTAL_MB = 60;            // Total ION memory
constexpr size_t ION_RESERVED_MB = 22;         // Reserved for H26X + ISP
constexpr size_t ION_AVAILABLE_MB = ION_TOTAL_MB - ION_RESERVED_MB;

} // namespace node
