#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "cv_types.h"

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_buffer.h>
#include <cvi_comm_vb.h>
#endif

namespace lua_cv {

#ifdef USE_CVI_MPI
class VbPoolPlan {
public:
    struct Pool {
        uint32_t width = 0;
        uint32_t height = 0;
        PixelFormat format = PixelFormat::UNKNOWN;
        uint32_t block_count = 0;
        VB_REMAP_MODE_E remap = VB_REMAP_MODE_CACHED;
        uint32_t block_size = 0;
        VbPoolUsage usage = VbPoolUsage::ANY;
    };

    void add_pool(uint32_t width, uint32_t height, PixelFormat format,
                  uint32_t block_count, VB_REMAP_MODE_E remap,
                  VbPoolUsage usage = VbPoolUsage::ANY);

    bool build();
    const VB_CONFIG_S& vb_config() const { return vb_config_; }
    const std::vector<Pool>& pools() const { return pools_; }

    VB_POOL find_pool(uint32_t width, uint32_t height, PixelFormat format,
                      VbPoolUsage usage = VbPoolUsage::ANY) const;
    uint32_t find_block_size(uint32_t width, uint32_t height, PixelFormat format) const;

private:
    std::vector<Pool> pools_;
    VB_CONFIG_S vb_config_{};
    bool built_ = false;
};

class MmfContext {
public:
    struct ModePlan {
        VI_VPSS_MODE_S vi_vpss_mode{};
        VPSS_MODE_S vpss_mode{};
    };

    struct Config {
        VbPoolPlan vb_plan;
        ModePlan mode_plan;
        bool force_reset = false;
    };

    static MmfContext& instance();
    static bool build_default_config(Config* config);
    static bool is_supported_camera_size(uint32_t width, uint32_t height);
    static bool is_supported_image_size(uint32_t width, uint32_t height);
    static int camera_vi_pipe();
    static int camera_vi_pool();
    static int camera_vpss_pool();
    static int vpss_group_for_camera();
    static int vpss_group_for_mem();
    static int vpss_channel_for_camera();
    static int vpss_channel_for_camera_stream();
    static int vpss_channel_for_mem();
    static uint32_t camera_stream_width();
    static uint32_t camera_stream_height();
    static PixelFormat camera_stream_format();
    static uint32_t camera_stream_depth();
    static uint32_t camera_infer_width();
    static uint32_t camera_infer_height();
    static PixelFormat camera_infer_format();
    static uint32_t camera_infer_depth();
    static int vpss_channel_for_camera_preview();
    static uint32_t camera_preview_width();
    static uint32_t camera_preview_height();
    static PixelFormat camera_preview_format();
    static uint32_t camera_preview_depth();
    static uint32_t vpss_max_width_for_camera();
    static uint32_t vpss_max_height_for_camera();
    static uint32_t vpss_max_width_for_mem();
    static uint32_t vpss_max_height_for_mem();
    static int vpss_dev_for_camera();
    static int vpss_dev_for_mem();

    bool init(const Config& config);
    void shutdown();
    bool is_initialized() const { return initialized_; }

    const VbPoolPlan& vb_plan() const { return vb_plan_; }
    const ModePlan& mode_plan() const { return mode_plan_; }

    void dump_status(const char* tag) const;

    void acquire_mem_vpss_group(uint32_t input_width, uint32_t input_height, PixelFormat input_format);
    void release_mem_vpss_group();
    bool mem_vpss_group_created() const { return mem_group_created_; }
    bool mem_vpss_group_started() const { return mem_group_started_; }
    void set_mem_vpss_group_started(bool v) { mem_group_started_ = v; }

    uint32_t mem_vpss_chn_out_width() const { return mem_chn_out_width_; }
    uint32_t mem_vpss_chn_out_height() const { return mem_chn_out_height_; }
    PixelFormat mem_vpss_chn_out_format() const { return mem_chn_out_format_; }
    bool mem_vpss_chn_letterbox() const { return mem_chn_letterbox_; }
    uint8_t mem_vpss_chn_pad_value() const { return mem_chn_pad_value_; }
    void set_mem_vpss_chn_state(uint32_t w, uint32_t h, PixelFormat fmt,
                                bool letterbox, uint8_t pad_value) {
        mem_chn_out_width_ = w;
        mem_chn_out_height_ = h;
        mem_chn_out_format_ = fmt;
        mem_chn_letterbox_ = letterbox;
        mem_chn_pad_value_ = pad_value;
    }

private:
    MmfContext() = default;
    void resolve_vpss_groups();

    bool initialized_ = false;
    VbPoolPlan vb_plan_;
    ModePlan mode_plan_;
    int resolved_mem_group_ = -1;
    int resolved_isp_group_ = -1;
    bool groups_resolved_ = false;

    int mem_group_refcount_ = 0;
    bool mem_group_created_ = false;
    bool mem_group_started_ = false;
    uint32_t mem_chn_out_width_ = 0;
    uint32_t mem_chn_out_height_ = 0;
    PixelFormat mem_chn_out_format_ = PixelFormat::UNKNOWN;
    bool mem_chn_letterbox_ = false;
    uint8_t mem_chn_pad_value_ = 0;
    std::mutex mem_group_mutex_;
};
#endif

} // namespace lua_cv
