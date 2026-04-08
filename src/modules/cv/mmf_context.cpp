#include "mmf_context.h"

#include <array>
#include <cstring>
#include <iostream>

namespace lua_cv {

#ifdef USE_CVI_MPI
#include <cvi_vpss.h>

namespace {
struct Resolution {
    uint32_t width = 0;
    uint32_t height = 0;
};

enum class VpssInputKind {
    Mem,
    Isp,
};

struct VpssGroupPlan {
    int group = -1;
    int vpss_dev = -1;
    int channel = 0;
    VpssInputKind input = VpssInputKind::Mem;
    uint32_t max_width = 0;
    uint32_t max_height = 0;
};

struct CameraOutputPlan {
    int channel = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::UNKNOWN;
    uint32_t depth = 0;
};

struct VbPoolSpec {
    uint32_t width = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::UNKNOWN;
    uint32_t block_count = 0;
    bool cached = true;
    VbPoolUsage usage = VbPoolUsage::ANY;
};

struct MmfPlan {
    int camera_vi_pipe = 0;
    int pool_camera_vi = 0;
    int pool_camera_vpss = 0;
    std::array<VpssGroupPlan, 2> groups{};
    CameraOutputPlan camera_stream{};
    CameraOutputPlan camera_infer{};
    CameraOutputPlan camera_preview{};  // Chn2: NV21 downscaled for JPEG preview
    std::array<Resolution, 3> camera_sizes{};
    std::array<Resolution, 3> image_sizes{};
    std::array<VbPoolSpec, 6> vb_pools{};
};

const MmfPlan& plan() {
    static constexpr MmfPlan kPlan = {
        0,
        0,
        1,
        {
            VpssGroupPlan{0, 1, 1, VpssInputKind::Isp, 1920, 1080},
            VpssGroupPlan{5, 0, 0, VpssInputKind::Mem, 1280, 720},
        },
        CameraOutputPlan{0, 1920, 1080, PixelFormat::NV21, 2},   // camera_stream: Chn0 (depth=2: 3 blocks = 2 queue + 1 VENC)
        CameraOutputPlan{1, 640, 640, PixelFormat::RGB, 2},       // camera_infer:  Chn1
        CameraOutputPlan{2, 1280, 720, PixelFormat::NV21, 2},      // camera_preview: Chn2 → VENC MJPEG (depth=2: VPSS queues 2 + 1 in VENC + 1 margin)
        {
            Resolution{1920, 1080},
            Resolution{0, 0},
            Resolution{0, 0},
        },
        {
            Resolution{640, 640},
            Resolution{0, 0},
            Resolution{0, 0},
        },
        {
            VbPoolSpec{1920, 1080, PixelFormat::NV21, 3, true, VbPoolUsage::CAMERA_VI},        // Pool 0: VI input (depth=2+1margin=3)
            VbPoolSpec{1920, 1080, PixelFormat::NV21, 3, true, VbPoolUsage::CAMERA_STREAM},   // Pool 1: VPSS Chn0 stream (depth=2+1VENC=3)
            VbPoolSpec{1280, 720, PixelFormat::NV21, 3, true, VbPoolUsage::CAMERA_PREVIEW},   // Pool 2: VPSS Chn2 preview (depth=2+VENC=3 blocks minimum)
            VbPoolSpec{640, 640, PixelFormat::RGB, 2, true, VbPoolUsage::VPSS_PREPROCESS},     // Pool 3: VPSS Grp5 preprocess output
            VbPoolSpec{640, 640, PixelFormat::RGB, 3, true, VbPoolUsage::CAMERA_INFER},        // Pool 4: VPSS Chn1 infer (depth=2+hold=1 = 3)
            VbPoolSpec{0, 0, PixelFormat::UNKNOWN, 0, true, VbPoolUsage::ANY},
        },
    };
    return kPlan;
}

template <std::size_t N>
bool is_supported_size(const std::array<Resolution, N>& sizes,
                       uint32_t width,
                       uint32_t height) {
    for (const auto& r : sizes) {
        if (r.width == 0 || r.height == 0) {
            continue;
        }
        if (r.width == width && r.height == height) {
            return true;
        }
    }
    return false;
}

int vpss_group_for_input(VpssInputKind input) {
    for (const auto& group : plan().groups) {
        if (group.input == input) {
            return group.group;
        }
    }
    return -1;
}

int vpss_dev_for_input(VpssInputKind input) {
    for (const auto& group : plan().groups) {
        if (group.input == input) {
            return group.vpss_dev;
        }
    }
    return -1;
}

int vpss_channel_for_input(VpssInputKind input) {
    for (const auto& group : plan().groups) {
        if (group.input == input) {
            return group.channel;
        }
    }
    return -1;
}

uint32_t vpss_max_width_for_input(VpssInputKind input) {
    for (const auto& group : plan().groups) {
        if (group.input == input) {
            return group.max_width;
        }
    }
    return 0;
}

uint32_t vpss_max_height_for_input(VpssInputKind input) {
    for (const auto& group : plan().groups) {
        if (group.input == input) {
            return group.max_height;
        }
    }
    return 0;
}

bool is_supported_mem_size(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) {
        return false;
    }
    if (width < height) {
        return false;
    }
    const uint32_t max_width = vpss_max_width_for_input(VpssInputKind::Mem);
    const uint32_t max_height = vpss_max_height_for_input(VpssInputKind::Mem);
    if (max_width == 0 || max_height == 0) {
        return false;
    }
    return width <= max_width && height <= max_height;
}

bool build_mode_plan(MmfContext::ModePlan* mode_plan) {
    if (!mode_plan) {
        return false;
    }
    std::memset(mode_plan, 0, sizeof(*mode_plan));

    for (int i = 0; i < VI_MAX_PIPE_NUM; ++i) {
        mode_plan->vi_vpss_mode.aenMode[i] = VI_OFFLINE_VPSS_ONLINE;
    }

    mode_plan->vpss_mode.enMode = VPSS_MODE_DUAL;
    const int default_pipe = plan().camera_vi_pipe;
    for (int i = 0; i < VPSS_IP_NUM; ++i) {
        mode_plan->vpss_mode.aenInput[i] = VPSS_INPUT_MEM;
        mode_plan->vpss_mode.ViPipe[i] = default_pipe;
    }

    const int mem_dev = vpss_dev_for_input(VpssInputKind::Mem);
    const int isp_dev = vpss_dev_for_input(VpssInputKind::Isp);
    if (mem_dev < 0 || mem_dev >= VPSS_IP_NUM || isp_dev < 0 || isp_dev >= VPSS_IP_NUM) {
        return false;
    }

    // VI_OFFLINE_VPSS_ONLINE mode requires:
    // - ISP dev: VPSS_INPUT_ISP (receives data directly from ISP pipeline)
    // - MEM dev: VPSS_INPUT_MEM (receives data via SendFrame for offline processing)
    mode_plan->vpss_mode.aenInput[mem_dev] = VPSS_INPUT_MEM;
    mode_plan->vpss_mode.ViPipe[mem_dev] = default_pipe;
    mode_plan->vpss_mode.aenInput[isp_dev] = VPSS_INPUT_ISP;
    mode_plan->vpss_mode.ViPipe[isp_dev] = default_pipe;
    return true;
}

bool build_vb_plan(VbPoolPlan* vb_plan) {
    if (!vb_plan) {
        return false;
    }

    for (const auto& pool : plan().vb_pools) {
        if (pool.width == 0 || pool.height == 0 || pool.block_count == 0) {
            continue;
        }
        vb_plan->add_pool(pool.width,
                          pool.height,
                          pool.format,
                          pool.block_count,
                          pool.cached ? VB_REMAP_MODE_CACHED : VB_REMAP_MODE_NOCACHE,
                          pool.usage);
    }
    return true;
}
}  // namespace

void VbPoolPlan::add_pool(uint32_t width, uint32_t height, PixelFormat format,
                          uint32_t block_count, VB_REMAP_MODE_E remap,
                          VbPoolUsage usage) {
    Pool pool;
    pool.width = width;
    pool.height = height;
    pool.format = format;
    pool.block_count = block_count;
    pool.remap = remap;
    pool.usage = usage;
    pools_.push_back(pool);
}

bool VbPoolPlan::build() {
    std::memset(&vb_config_, 0, sizeof(vb_config_));
    vb_config_.u32MaxPoolCnt = static_cast<CVI_U32>(pools_.size());

    for (size_t i = 0; i < pools_.size(); ++i) {
        Pool& pool = pools_[i];
        PIXEL_FORMAT_E fmt = to_cvi_pixel_format(pool.format);
        if (fmt == PIXEL_FORMAT_MAX) {
            std::cerr << "[ERROR] VbPoolPlan: invalid pixel format" << std::endl;
            return false;
        }

        pool.block_size = COMMON_GetPicBufferSize(pool.width, pool.height, fmt,
                                                  DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
        vb_config_.astCommPool[i].u32BlkSize = pool.block_size;
        vb_config_.astCommPool[i].u32BlkCnt = pool.block_count;
        vb_config_.astCommPool[i].enRemapMode = pool.remap;
        std::snprintf(vb_config_.astCommPool[i].acName,
                      sizeof(vb_config_.astCommPool[i].acName),
                      "pool_%zu", i);
    }

    built_ = true;
    return true;
}

VB_POOL VbPoolPlan::find_pool(uint32_t width, uint32_t height, PixelFormat format,
                             VbPoolUsage usage) const {
    if (!built_ || pools_.empty()) {
        return VB_INVALID_POOLID;
    }

    auto select_pool = [&](PixelFormat target_format) -> VB_POOL {
        uint32_t needed = find_block_size(width, height, target_format);
        if (needed == 0) {
            return VB_INVALID_POOLID;
        }

        VB_POOL best_pool = VB_INVALID_POOLID;
        uint32_t best_size = 0;

        for (size_t i = 0; i < pools_.size(); ++i) {
            const Pool& pool = pools_[i];
            if (pool.format != target_format || pool.block_size == 0) {
                continue;
            }
            // Strict usage filtering: when usage != ANY, only match pools
            // with the same usage tag. When usage == ANY, match any pool.
            if (usage != VbPoolUsage::ANY && pool.usage != usage) {
                continue;
            }
            if (pool.block_size >= needed &&
                (best_pool == VB_INVALID_POOLID || pool.block_size < best_size)) {
                best_pool = static_cast<VB_POOL>(i);
                best_size = pool.block_size;
            }
        }
        return best_pool;
    };

    struct Candidate {
        VB_POOL id = VB_INVALID_POOLID;
        uint32_t block_size = 0;
        uint32_t block_count = 0;
    };

    auto make_candidate = [&](PixelFormat target_format) -> Candidate {
        VB_POOL pool_id = select_pool(target_format);
        if (pool_id == VB_INVALID_POOLID) {
            return {};
        }
        const auto& pool = pools_[static_cast<size_t>(pool_id)];
        return {pool_id, pool.block_size, pool.block_count};
    };

    Candidate primary = make_candidate(format);
    Candidate fallback{};
    if (format == PixelFormat::RGB) {
        fallback = make_candidate(PixelFormat::BGR);
    } else if (format == PixelFormat::BGR) {
        fallback = make_candidate(PixelFormat::RGB);
    }

    if (primary.id == VB_INVALID_POOLID) {
        return fallback.id;
    }
    if (fallback.id == VB_INVALID_POOLID) {
        return primary.id;
    }

    if (fallback.block_size < primary.block_size) {
        return fallback.id;
    }
    if (fallback.block_size > primary.block_size) {
        return primary.id;
    }
    if (fallback.block_count > primary.block_count) {
        return fallback.id;
    }
    return primary.id;
}

uint32_t VbPoolPlan::find_block_size(uint32_t width, uint32_t height, PixelFormat format) const {
    PIXEL_FORMAT_E fmt = to_cvi_pixel_format(format);
    if (fmt == PIXEL_FORMAT_MAX) {
        return 0;
    }
    return COMMON_GetPicBufferSize(width, height, fmt, DATA_BITWIDTH_8,
                                   COMPRESS_MODE_NONE, 0);
}

MmfContext& MmfContext::instance() {
    static MmfContext ctx;
    return ctx;
}

bool MmfContext::build_default_config(Config* config) {
    if (!config) {
        return false;
    }
    config->vb_plan = VbPoolPlan();
    if (!build_vb_plan(&config->vb_plan)) {
        return false;
    }
    if (!build_mode_plan(&config->mode_plan)) {
        return false;
    }
    return true;
}

bool MmfContext::is_supported_camera_size(uint32_t width, uint32_t height) {
    return is_supported_size(plan().camera_sizes, width, height);
}

bool MmfContext::is_supported_image_size(uint32_t width, uint32_t height) {
    return is_supported_mem_size(width, height);
}

int MmfContext::camera_vi_pipe() {
    return plan().camera_vi_pipe;
}

int MmfContext::camera_vi_pool() {
    return plan().pool_camera_vi;
}

int MmfContext::camera_vpss_pool() {
    return plan().pool_camera_vpss;
}

int MmfContext::vpss_group_for_camera() {
    MmfContext& ctx = MmfContext::instance();
    if (ctx.initialized_ && ctx.groups_resolved_ && ctx.resolved_isp_group_ >= 0) {
        return ctx.resolved_isp_group_;
    }
    return vpss_group_for_input(VpssInputKind::Isp);
}

int MmfContext::vpss_group_for_mem() {
    MmfContext& ctx = MmfContext::instance();
    if (ctx.initialized_ && ctx.groups_resolved_ && ctx.resolved_mem_group_ >= 0) {
        return ctx.resolved_mem_group_;
    }
    return vpss_group_for_input(VpssInputKind::Mem);
}

int MmfContext::vpss_channel_for_camera() {
    return vpss_channel_for_input(VpssInputKind::Isp);
}

int MmfContext::vpss_channel_for_camera_stream() {
    return plan().camera_stream.channel;
}

int MmfContext::vpss_channel_for_mem() {
    return vpss_channel_for_input(VpssInputKind::Mem);
}

uint32_t MmfContext::camera_stream_width() {
    return plan().camera_stream.width;
}

uint32_t MmfContext::camera_stream_height() {
    return plan().camera_stream.height;
}

PixelFormat MmfContext::camera_stream_format() {
    return plan().camera_stream.format;
}

uint32_t MmfContext::camera_stream_depth() {
    return plan().camera_stream.depth;
}

uint32_t MmfContext::camera_infer_width() {
    return plan().camera_infer.width;
}

uint32_t MmfContext::camera_infer_height() {
    return plan().camera_infer.height;
}

PixelFormat MmfContext::camera_infer_format() {
    return plan().camera_infer.format;
}

uint32_t MmfContext::camera_infer_depth() {
    return plan().camera_infer.depth;
}

int MmfContext::vpss_channel_for_camera_preview() {
    return plan().camera_preview.channel;
}

uint32_t MmfContext::camera_preview_width() {
    return plan().camera_preview.width;
}

uint32_t MmfContext::camera_preview_height() {
    return plan().camera_preview.height;
}

PixelFormat MmfContext::camera_preview_format() {
    return plan().camera_preview.format;
}

uint32_t MmfContext::camera_preview_depth() {
    return plan().camera_preview.depth;
}

uint32_t MmfContext::vpss_max_width_for_camera() {
    return vpss_max_width_for_input(VpssInputKind::Isp);
}

uint32_t MmfContext::vpss_max_height_for_camera() {
    return vpss_max_height_for_input(VpssInputKind::Isp);
}

uint32_t MmfContext::vpss_max_width_for_mem() {
    return vpss_max_width_for_input(VpssInputKind::Mem);
}

uint32_t MmfContext::vpss_max_height_for_mem() {
    return vpss_max_height_for_input(VpssInputKind::Mem);
}

int MmfContext::vpss_dev_for_camera() {
    return vpss_dev_for_input(VpssInputKind::Isp);
}

int MmfContext::vpss_dev_for_mem() {
    return vpss_dev_for_input(VpssInputKind::Mem);
}

bool MmfContext::init(const Config& config) {
    if (initialized_) {
        return true;
    }

    if (config.force_reset) {
        CVI_SYS_Exit();
        CVI_VB_Exit();
    }

    vb_plan_ = config.vb_plan;
    if (!vb_plan_.build()) {
        return false;
    }

    CVI_S32 rc = CVI_VB_SetConfig(&vb_plan_.vb_config());
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_SetConfig failed: " << rc << std::endl;
        return false;
    }

    rc = CVI_VB_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_VB_Init failed: " << rc << std::endl;
        return false;
    }

    rc = CVI_SYS_Init();
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_Init failed: " << rc << std::endl;
        CVI_VB_Exit();
        return false;
    }

    mode_plan_ = config.mode_plan;

    rc = CVI_SYS_SetVIVPSSMode(&mode_plan_.vi_vpss_mode);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_SetVIVPSSMode failed: " << rc << std::endl;
        CVI_SYS_Exit();
        CVI_VB_Exit();
        return false;
    }

    rc = CVI_SYS_SetVPSSModeEx(&mode_plan_.vpss_mode);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[ERROR] CVI_SYS_SetVPSSModeEx failed: " << rc << std::endl;
        CVI_SYS_Exit();
        CVI_VB_Exit();
        return false;
    }

    resolve_vpss_groups();

    std::cout << "[MMF] VB pools configured:" << std::endl;
    for (size_t i = 0; i < vb_plan_.pools().size(); ++i) {
        const auto& pool = vb_plan_.pools()[i];
        std::cout << "  Pool " << i << ": " << pool.width << "x" << pool.height
                  << " " << pixel_format_name(pool.format)
                  << " size=" << pool.block_size
                  << " count=" << pool.block_count << std::endl;
    }

    initialized_ = true;
    return true;
}

void MmfContext::shutdown() {
    if (!initialized_) {
        return;
    }
    CVI_SYS_Exit();
    CVI_VB_Exit();
    initialized_ = false;
    groups_resolved_ = false;
    resolved_mem_group_ = -1;
    resolved_isp_group_ = -1;
}

void MmfContext::dump_status(const char* tag) const {
    std::cout << "\n[MMF] Status dump: " << (tag ? tag : "") << std::endl;
    std::system("cat /proc/cvitek/sys | grep -A 10 'BIND RELATION'");
    std::system("cat /proc/cvitek/vi | grep -A 5 'VI CHN STATUS'");
    std::system("cat /proc/cvitek/vpss | grep -A 20 'WORK STATUS'");
    std::system("cat /proc/cvitek/vb | grep -A 20 'PoolId.*0'");
}

void MmfContext::resolve_vpss_groups() {
    resolved_isp_group_ = vpss_group_for_input(VpssInputKind::Isp);
    resolved_mem_group_ = vpss_group_for_input(VpssInputKind::Mem);

    if (resolved_isp_group_ < 0) {
        resolved_isp_group_ = 0;
    }
    if (resolved_mem_group_ < 0 || resolved_mem_group_ == resolved_isp_group_) {
        resolved_mem_group_ = (resolved_isp_group_ == 0) ? 1 : 0;
    }

    groups_resolved_ = true;
    std::cout << "[MMF] VPSS groups resolved: isp=" << resolved_isp_group_
              << " mem=" << resolved_mem_group_ << std::endl;
}
#endif

} // namespace lua_cv
