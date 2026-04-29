#include "cvi_vpss_processor.h"

#include <stdexcept>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <fstream>

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <linux/cvi_comm_vpss.h>
#include <linux/cvi_errno.h>
#endif

namespace lua_cv {
#ifdef USE_CVI_MPI
namespace {
int select_vpss_group_for_mem() {
    return MmfContext::vpss_group_for_mem();
}

void dump_vb_proc(const char* tag) {
    std::ifstream file("/proc/cvitek/vb");
    if (!file) {
        std::cerr << "[VBTRACE] " << tag << " (cannot open /proc/cvitek/vb)" << std::endl;
        return;
    }

    std::ostringstream oss;
    std::string line;
    size_t total = 0;
    const size_t max_bytes = 4096;
    while (std::getline(file, line)) {
        if (total + line.size() + 1 > max_bytes) {
            line.resize(max_bytes - total);
        }
        oss << line << "\n";
        total += line.size() + 1;
        if (total >= max_bytes) {
            break;
        }
    }
    std::cerr << "[VBTRACE] " << tag << "\n" << oss.str() << std::endl;
}

void log_vpss_create_context(const VPSS_GRP_ATTR_S& grp_attr,
                             PixelFormat input_format,
                             VPSS_GRP grp,
                             CVI_S32 rc) {
    std::cerr << "[ERROR] CviVpssProcessor: CVI_VPSS_CreateGrp failed: 0x"
              << std::hex << rc << std::dec
              << " grp=" << grp
              << " vpss_dev=" << static_cast<int>(grp_attr.u8VpssDev)
              << " input=" << pixel_format_name(input_format)
              << "(" << static_cast<int>(grp_attr.enPixelFormat) << ")"
              << " max=" << grp_attr.u32MaxW << "x" << grp_attr.u32MaxH
              << std::endl;
    VPSS_GRP avail_grp = CVI_VPSS_GetAvailableGrp();
    std::cerr << "[ERROR] CviVpssProcessor: VPSS available grp=" << avail_grp << std::endl;

    VPSS_MODE_S vpss_mode{};
    CVI_S32 mode_rc = CVI_SYS_GetVPSSModeEx(&vpss_mode);
    if (mode_rc == CVI_SUCCESS) {
        std::ostringstream oss;
        oss << "[ERROR] CviVpssProcessor: VPSS mode ex enMode=" << vpss_mode.enMode;
        for (int i = 0; i < VPSS_IP_NUM; ++i) {
            oss << " input" << i << "=" << vpss_mode.aenInput[i]
                << " viPipe" << i << "=" << vpss_mode.ViPipe[i];
        }
        std::cerr << oss.str() << std::endl;
    } else {
        std::cerr << "[WARN] CviVpssProcessor: CVI_SYS_GetVPSSModeEx failed: 0x"
                  << std::hex << mode_rc << std::dec << std::endl;
    }

    VI_VPSS_MODE_S vi_vpss{};
    CVI_S32 vi_mode_rc = CVI_SYS_GetVIVPSSMode(&vi_vpss);
    if (vi_mode_rc == CVI_SUCCESS) {
        std::ostringstream oss;
        oss << "[ERROR] CviVpssProcessor: VI-VPSS mode";
        for (int i = 0; i < VI_MAX_PIPE_NUM; ++i) {
            oss << " pipe" << i << "=" << vi_vpss.aenMode[i];
            if (i >= 3) {
                break;
            }
        }
        std::cerr << oss.str() << std::endl;
    } else {
        std::cerr << "[WARN] CviVpssProcessor: CVI_SYS_GetVIVPSSMode failed: 0x"
                  << std::hex << vi_mode_rc << std::dec << std::endl;
    }
}
}  // namespace
#endif

CviVpssProcessor::CviVpssProcessor() {
#ifdef USE_CVI_MPI
    chn_ = static_cast<VPSS_CHN>(MmfContext::vpss_channel_for_mem());
#endif
}

CviVpssProcessor::~CviVpssProcessor() {
#ifdef USE_CVI_MPI
    // Only destroy group if it was created and not already destroyed
    if (group_created_) {
        destroy_group();
    }
#endif
}

void CviVpssProcessor::resize(Frame& frame, int width, int height) {
#ifdef USE_CVI_MPI
    last_error_ = CVI_SUCCESS;
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::resize - frame is empty");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("CviVpssProcessor::resize - invalid dimensions");
    }

    // Check against VPSS MEM group capacity, not current frame size
    // This allows upscaling within VPSS hardware limits and available VB pools
    const uint32_t max_w = MmfContext::vpss_max_width_for_mem();
    const uint32_t max_h = MmfContext::vpss_max_height_for_mem();
    if (static_cast<uint32_t>(width) > max_w || static_cast<uint32_t>(height) > max_h) {
        throw std::invalid_argument("CviVpssProcessor::resize - exceeds VPSS MEM capacity");
    }

    PixelFormat input_format = frame.pixel_format();
    if (input_format == PixelFormat::UNKNOWN) {
        input_format = PixelFormat::BGR;
    }

    process_frame(frame, static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                  input_format, false, 0, 0, 0, 0, false, 0);
#else
    (void)frame;
    (void)width;
    (void)height;
    throw std::runtime_error("CviVpssProcessor requires USE_CVI_MPI");
#endif
}

void CviVpssProcessor::cvtColor(Frame& frame, ColorConversion code) {
#ifdef USE_CVI_MPI
    last_error_ = CVI_SUCCESS;
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::cvtColor - frame is empty");
    }

    PixelFormat input_format = frame.pixel_format();
    if (input_format == PixelFormat::UNKNOWN) {
        input_format = PixelFormat::BGR;
    }

    PixelFormat output_format = input_format;
    switch (code) {
        case ColorConversion::BGR2RGB:
        case ColorConversion::GRAY2RGB:
            output_format = PixelFormat::RGB;
            break;
        case ColorConversion::RGB2BGR:
        case ColorConversion::GRAY2BGR:
            output_format = PixelFormat::BGR;
            break;
        case ColorConversion::BGR2GRAY:
        case ColorConversion::RGB2GRAY:
            output_format = PixelFormat::GRAY;
            break;
        default:
            throw std::invalid_argument("CviVpssProcessor::cvtColor - unsupported conversion");
    }

    process_frame(frame, static_cast<uint32_t>(frame.width()),
                  static_cast<uint32_t>(frame.height()), output_format,
                  false, 0, 0, 0, 0, false, 0);
#else
    (void)frame;
    (void)code;
    throw std::runtime_error("CviVpssProcessor requires USE_CVI_MPI");
#endif
}

void CviVpssProcessor::convert_format(Frame& frame, PixelFormat output_format) {
#ifdef USE_CVI_MPI
    last_error_ = CVI_SUCCESS;
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::convert_format - frame is empty");
    }
    if (output_format == PixelFormat::UNKNOWN) {
        throw std::invalid_argument("CviVpssProcessor::convert_format - output format is unknown");
    }
    if (frame.pixel_format() == output_format) {
        return;
    }

    PixelFormat input_format = frame.pixel_format();
    if (input_format == PixelFormat::UNKNOWN) {
        input_format = PixelFormat::BGR;
    }

    process_frame(frame, static_cast<uint32_t>(frame.width()),
                  static_cast<uint32_t>(frame.height()), output_format,
                  false, 0, 0, 0, 0, false, 0);
#else
    (void)frame;
    (void)output_format;
    throw std::runtime_error("CviVpssProcessor requires USE_CVI_MPI");
#endif
}

void CviVpssProcessor::crop(Frame& frame, int x, int y, int w, int h) {
#ifdef USE_CVI_MPI
    last_error_ = CVI_SUCCESS;
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::crop - frame is empty");
    }
    if (x < 0 || y < 0 || w <= 0 || h <= 0 ||
        x + w > frame.width() || y + h > frame.height()) {
        throw std::invalid_argument("CviVpssProcessor::crop - invalid crop region");
    }

    PixelFormat input_format = frame.pixel_format();
    if (input_format == PixelFormat::UNKNOWN) {
        input_format = PixelFormat::BGR;
    }

    process_frame(frame, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                  input_format, true, x, y, w, h, false, 0);
#else
    (void)frame;
    (void)x;
    (void)y;
    (void)w;
    (void)h;
    throw std::runtime_error("CviVpssProcessor requires USE_CVI_MPI");
#endif
}

void CviVpssProcessor::crop_resize(Frame& frame,
                                    int crop_x, int crop_y, int crop_w, int crop_h,
                                    int out_width, int out_height,
                                    PixelFormat output_format) {
#ifdef USE_CVI_MPI
    last_error_ = CVI_SUCCESS;
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::crop_resize - frame is empty");
    }
    if (crop_x < 0 || crop_y < 0 || crop_w <= 0 || crop_h <= 0 ||
        crop_x + crop_w > frame.width() || crop_y + crop_h > frame.height()) {
        throw std::invalid_argument("CviVpssProcessor::crop_resize - invalid crop region");
    }
    if (out_width <= 0 || out_height <= 0) {
        throw std::invalid_argument("CviVpssProcessor::crop_resize - invalid output dimensions");
    }

    // Check against VPSS MEM group capacity
    const uint32_t max_w = MmfContext::vpss_max_width_for_mem();
    const uint32_t max_h = MmfContext::vpss_max_height_for_mem();
    if (static_cast<uint32_t>(out_width) > max_w || static_cast<uint32_t>(out_height) > max_h) {
        throw std::invalid_argument("CviVpssProcessor::crop_resize - output exceeds VPSS MEM capacity");
    }

    PixelFormat input_format = frame.pixel_format();
    if (input_format == PixelFormat::UNKNOWN) {
        input_format = PixelFormat::BGR;
    }

    // Use default output format if not specified
    if (output_format == PixelFormat::UNKNOWN) {
        output_format = PixelFormat::RGB;
    }

    // Combined crop + resize in single VPSS pass
    process_frame(frame,
                  static_cast<uint32_t>(out_width), static_cast<uint32_t>(out_height),
                  output_format, true, crop_x, crop_y, crop_w, crop_h, false, 0);
#else
    (void)frame;
    (void)crop_x;
    (void)crop_y;
    (void)crop_w;
    (void)crop_h;
    (void)out_width;
    (void)out_height;
    (void)output_format;
    throw std::runtime_error("CviVpssProcessor requires USE_CVI_MPI");
#endif
}

void CviVpssProcessor::letterbox(Frame& frame, int width, int height, uint8_t pad_value,
                                 LetterboxMeta* meta, PixelFormat output_format) {
#ifdef USE_CVI_MPI
    last_error_ = CVI_SUCCESS;
    if (frame.empty()) {
        throw std::invalid_argument("CviVpssProcessor::letterbox - frame is empty");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("CviVpssProcessor::letterbox - invalid dimensions");
    }

    const int ori_w = frame.width();
    const int ori_h = frame.height();

    process_frame(frame, static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                  output_format, false, 0, 0, 0, 0, true, pad_value);

    if (meta) {
        const float scale_w = static_cast<float>(width) / static_cast<float>(ori_w);
        const float scale_h = static_cast<float>(height) / static_cast<float>(ori_h);
        meta->scale = std::min(scale_w, scale_h);
        const int new_w = static_cast<int>(ori_w * meta->scale + 0.5f);
        const int new_h = static_cast<int>(ori_h * meta->scale + 0.5f);
        meta->pad_x = (width - new_w) / 2;
        meta->pad_y = (height - new_h) / 2;
        meta->ori_w = ori_w;
        meta->ori_h = ori_h;
    }
#else
    (void)frame;
    (void)width;
    (void)height;
    (void)pad_value;
    (void)meta;
    throw std::runtime_error("CviVpssProcessor requires USE_CVI_MPI");
#endif
}

#ifdef USE_CVI_MPI
VIDEO_FRAME_INFO_S CviVpssProcessor::mat_to_video_frame(const cv::Mat& mat, VB_BLK& vb_block) {
    return mat_to_video_frame(mat, vb_block, VB_INVALID_POOLID);
}

VIDEO_FRAME_INFO_S CviVpssProcessor::mat_to_video_frame(const cv::Mat& mat, VB_BLK& vb_block, VB_POOL pool) {
    if (mat.empty()) {
        throw std::invalid_argument("mat_to_video_frame - input mat is empty");
    }

    PixelFormat fmt = (mat.channels() == 1) ? PixelFormat::GRAY : PixelFormat::BGR;
    PIXEL_FORMAT_E cvi_fmt = to_cvi_pixel_format(fmt);
    if (cvi_fmt == PIXEL_FORMAT_MAX) {
        throw std::invalid_argument("mat_to_video_frame - unsupported format");
    }

    VB_CAL_CONFIG_S cal{};
    COMMON_GetPicBufferConfig(static_cast<CVI_U32>(mat.cols),
                              static_cast<CVI_U32>(mat.rows),
                              cvi_fmt, DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0, &cal);

    CVI_U32 alloc_size = cal.u32VBSize;
    auto try_get_block = [&](VB_POOL pool_id, CVI_U32 size) -> VB_BLK {
        VB_BLK blk = CVI_VB_GetBlock(pool_id, size);
        if (blk != VB_INVALID_HANDLE) {
            alloc_size = size;
        }
        return blk;
    };

    vb_block = try_get_block(pool, cal.u32VBSize);
    if (vb_block == VB_INVALID_HANDLE && pool != VB_INVALID_POOLID) {
        vb_block = try_get_block(VB_INVALID_POOLID, cal.u32VBSize);
    }
    if (vb_block == VB_INVALID_HANDLE) {
        for (int retry = 0; retry < 3 && vb_block == VB_INVALID_HANDLE; ++retry) {
            usleep(2000);
            vb_block = try_get_block(VB_INVALID_POOLID, cal.u32VBSize);
        }
    }
    if (vb_block == VB_INVALID_HANDLE) {
        VB_CONFIG_S vb_config{};
        CVI_S32 cfg_rc = CVI_VB_GetConfig(&vb_config);
        if (cfg_rc == CVI_SUCCESS && vb_config.u32MaxPoolCnt > 0) {
            struct Candidate {
                VB_POOL id;
                CVI_U32 blk_size;
            };
            std::vector<Candidate> candidates;
            candidates.reserve(vb_config.u32MaxPoolCnt);
            for (CVI_U32 i = 0; i < vb_config.u32MaxPoolCnt; ++i) {
                CVI_U32 blk_size = vb_config.astCommPool[i].u32BlkSize;
                CVI_U32 blk_cnt = vb_config.astCommPool[i].u32BlkCnt;
                if (blk_size >= cal.u32VBSize && blk_cnt > 0) {
                    candidates.push_back({static_cast<VB_POOL>(i), blk_size});
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.blk_size > b.blk_size;
                      });
            for (const auto& cand : candidates) {
                vb_block = try_get_block(cand.id, cal.u32VBSize);
                if (vb_block != VB_INVALID_HANDLE) {
                    break;
                }
                if (cand.blk_size != cal.u32VBSize) {
                    vb_block = try_get_block(cand.id, cand.blk_size);
                    if (vb_block != VB_INVALID_HANDLE) {
                        break;
                    }
                }
            }
        }
    }
    if (vb_block == VB_INVALID_HANDLE) {
        dump_vb_proc("mat_to_video_frame - CVI_VB_GetBlock failed");
        throw std::runtime_error("mat_to_video_frame - CVI_VB_GetBlock failed");
    }

    CVI_U64 phys = CVI_VB_Handle2PhysAddr(vb_block);
    void* virt = CVI_SYS_MmapCache(phys, alloc_size);
    if (!virt) {
        dump_vb_proc("mat_to_video_frame - CVI_SYS_MmapCache failed");
        CVI_VB_ReleaseBlock(vb_block);
        throw std::runtime_error("mat_to_video_frame - CVI_SYS_MmapCache failed");
    }

    uint8_t* dst = static_cast<uint8_t*>(virt);
    int copy_bytes = mat.cols * mat.channels();
    int dst_stride = static_cast<int>(cal.u32MainStride);

    for (int y = 0; y < mat.rows; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dst_stride,
                    mat.ptr<uint8_t>(y),
                    static_cast<size_t>(copy_bytes));
    }

    CVI_SYS_IonFlushCache(phys, virt, alloc_size);
    CVI_SYS_Munmap(virt, alloc_size);

    VIDEO_FRAME_INFO_S frame{};
    frame.stVFrame.u32Width = static_cast<CVI_U32>(mat.cols);
    frame.stVFrame.u32Height = static_cast<CVI_U32>(mat.rows);
    frame.stVFrame.enPixelFormat = cvi_fmt;
    frame.stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
    frame.stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    frame.stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
    frame.stVFrame.enColorGamut = COLOR_GAMUT_BT709;

    frame.stVFrame.u32Stride[0] = cal.u32MainStride;
    frame.stVFrame.u32Stride[1] = cal.u32CStride;
    frame.stVFrame.u32Stride[2] = cal.u32CStride;
    if (cal.plane_num <= 1) {
        frame.stVFrame.u32Length[0] = cal.u32MainSize;
        frame.stVFrame.u32Length[1] = 0;
        frame.stVFrame.u32Length[2] = 0;
    } else {
        frame.stVFrame.u32Length[0] = cal.u32MainYSize;
        frame.stVFrame.u32Length[1] = cal.u32MainCSize;
        frame.stVFrame.u32Length[2] = 0;
    }

    frame.stVFrame.u64PhyAddr[0] = phys;
    if (cal.u32MainCSize > 0) {
        frame.stVFrame.u64PhyAddr[1] = phys + cal.u32MainYSize;
    }

    frame.u32PoolId = CVI_VB_Handle2PoolId(vb_block);

    return frame;
}

int CviVpssProcessor::last_error() const {
#ifdef USE_CVI_MPI
    return last_error_;
#else
    return 0;
#endif
}

void CviVpssProcessor::ensure_group(uint32_t input_width, uint32_t input_height, PixelFormat input_format) {
    if (!MmfContext::instance().is_initialized()) {
        throw std::runtime_error("CviVpssProcessor - MmfContext not initialized");
    }

    const uint32_t max_width = MmfContext::vpss_max_width_for_mem();
    const uint32_t max_height = MmfContext::vpss_max_height_for_mem();
    if ((max_width > 0 && input_width > max_width) ||
        (max_height > 0 && input_height > max_height)) {
        std::ostringstream oss;
        oss << "CviVpssProcessor - input size exceeds mem VPSS max "
            << max_width << "x" << max_height
            << " (got " << input_width << "x" << input_height << ")";
        throw std::runtime_error(oss.str());
    }

    if (group_created_) {
        return;
    }

    MmfContext::instance().acquire_mem_vpss_group(input_width, input_height, input_format);
    grp_ = select_vpss_group_for_mem();

    group_created_ = true;
    group_started_ = false;
    max_width_ = input_width;
    max_height_ = input_height;
    input_format_ = input_format;
}

void CviVpssProcessor::ensure_channel(uint32_t out_width, uint32_t out_height, PixelFormat out_format,
                                      bool letterbox, uint8_t pad_value) {
    if (!group_created_) {
        throw std::runtime_error("CviVpssProcessor - VPSS group not created");
    }

    auto& ctx = MmfContext::instance();
    // Compare against the SHARED channel state (MmfContext), not per-instance state.
    // Multiple CviVpssProcessor instances share the same VPSS group+channel, so
    // when one processor reconfigures the channel (e.g., classify 224x224), others
    // (detectors 640x640) must detect the mismatch and reconfigure.
    bool need_reconfig = (!ctx.mem_vpss_group_started() ||
                          ctx.mem_vpss_chn_out_width() != out_width ||
                          ctx.mem_vpss_chn_out_height() != out_height ||
                          ctx.mem_vpss_chn_out_format() != out_format ||
                          ctx.mem_vpss_chn_letterbox() != letterbox ||
                          ctx.mem_vpss_chn_pad_value() != pad_value);

    if (!need_reconfig) {
        return;
    }

    // When the group is already running, a full StopGrp → ResetGrp cycle is required
    // to reinitialize the VPSS hardware pipeline before applying new channel attributes.
    if (ctx.mem_vpss_group_started()) {
        CVI_VPSS_DisableChn(grp_, chn_);
        CVI_S32 stop_rc = CVI_VPSS_StopGrp(grp_);
        if (stop_rc != CVI_SUCCESS) {
            std::cerr << "[VPSS-RECFG] StopGrp failed: 0x" << std::hex << stop_rc << std::dec << std::endl;
        }
        VPSS_GRP_ATTR_S grp_attr{};
        grp_attr.u32MaxW = max_width_;
        grp_attr.u32MaxH = max_height_;
        grp_attr.enPixelFormat = to_cvi_pixel_format(input_format_);
        grp_attr.u8VpssDev = static_cast<CVI_U8>(MmfContext::vpss_dev_for_mem());
        CVI_S32 attr_rc = CVI_VPSS_SetGrpAttr(grp_, &grp_attr);
        if (attr_rc != CVI_SUCCESS) {
            std::cerr << "[VPSS-RECFG] SetGrpAttr failed: 0x" << std::hex << attr_rc << std::dec << std::endl;
        }
        CVI_S32 reset_rc = CVI_VPSS_ResetGrp(grp_);
        if (reset_rc != CVI_SUCCESS) {
            std::cerr << "[VPSS-RECFG] ResetGrp failed: 0x" << std::hex << reset_rc << std::dec << std::endl;
        }
        ctx.set_mem_vpss_group_started(false);
    }

    CVI_VPSS_DisableChn(grp_, chn_);

    VPSS_CHN_ATTR_S chn_attr{};
    chn_attr.u32Width = out_width;
    chn_attr.u32Height = out_height;
    chn_attr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    chn_attr.enPixelFormat = to_cvi_pixel_format(out_format);
    chn_attr.stFrameRate.s32SrcFrameRate = -1;
    chn_attr.stFrameRate.s32DstFrameRate = -1;
    chn_attr.u32Depth = 1;
    if (letterbox) {
        chn_attr.stAspectRatio.enMode = ASPECT_RATIO_AUTO;
        chn_attr.stAspectRatio.bEnableBgColor = CVI_TRUE;
        chn_attr.stAspectRatio.u32BgColor =
            (static_cast<CVI_U32>(pad_value) << 16) |
            (static_cast<CVI_U32>(pad_value) << 8) |
            static_cast<CVI_U32>(pad_value);
    } else {
        chn_attr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
        chn_attr.stAspectRatio.bEnableBgColor = CVI_FALSE;
        chn_attr.stAspectRatio.u32BgColor = 0;
    }

    CVI_S32 rc = CVI_VPSS_SetChnAttr(grp_, chn_, &chn_attr);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviVpssProcessor - CVI_VPSS_SetChnAttr failed");
    }

    rc = CVI_VPSS_EnableChn(grp_, chn_);
    if (rc != CVI_SUCCESS) {
        throw std::runtime_error("CviVpssProcessor - CVI_VPSS_EnableChn failed");
    }

    if (!ctx.mem_vpss_group_started()) {
        rc = CVI_VPSS_StartGrp(grp_);
        if (rc != CVI_SUCCESS && rc != CVI_ERR_VPSS_EXIST) {
            throw std::runtime_error("CviVpssProcessor - CVI_VPSS_StartGrp failed");
        }
        ctx.set_mem_vpss_group_started(true);
    }

    VB_POOL pool = select_output_pool(out_width, out_height, out_format);
    if (pool == VB_INVALID_POOLID) {
        throw std::runtime_error("CviVpssProcessor - no suitable VB pool for output");
    }

    if (out_pool_ != VB_INVALID_POOLID && out_pool_ != pool) {
        CVI_VPSS_DetachVbPool(grp_, chn_);
    }
    if (out_pool_ != pool) {
        rc = CVI_VPSS_AttachVbPool(grp_, chn_, pool);
        if (rc != CVI_SUCCESS) {
            throw std::runtime_error("CviVpssProcessor - CVI_VPSS_AttachVbPool failed");
        }
        out_pool_ = pool;
    }

    // Update both per-instance and shared state
    out_width_ = out_width;
    out_height_ = out_height;
    out_format_ = out_format;
    letterbox_enabled_ = letterbox;
    pad_value_ = pad_value;
    ctx.set_mem_vpss_chn_state(out_width, out_height, out_format, letterbox, pad_value);
    group_started_ = true;
}

void CviVpssProcessor::process_frame(Frame& frame, uint32_t out_width, uint32_t out_height,
                                     PixelFormat out_format, bool use_crop,
                                     int crop_x, int crop_y, int crop_w, int crop_h,
                                     bool letterbox, uint8_t pad_value) {
    last_error_ = CVI_SUCCESS;
    PixelFormat input_format = frame.pixel_format();
    if (input_format == PixelFormat::UNKNOWN) {
        input_format = PixelFormat::BGR;
    }

    ensure_group(static_cast<uint32_t>(frame.width()), static_cast<uint32_t>(frame.height()), input_format);
    ensure_channel(out_width, out_height, out_format, letterbox, pad_value);

    VPSS_CROP_INFO_S crop_info{};
    if (use_crop) {
        crop_info.bEnable = CVI_TRUE;
        crop_info.enCropCoordinate = VPSS_CROP_ABS_COOR;
        crop_info.stCropRect.s32X = crop_x;
        crop_info.stCropRect.s32Y = crop_y;
        crop_info.stCropRect.u32Width = static_cast<CVI_U32>(crop_w);
        crop_info.stCropRect.u32Height = static_cast<CVI_U32>(crop_h);
        CVI_S32 rc = CVI_VPSS_SetChnCrop(grp_, chn_, &crop_info);
        if (rc != CVI_SUCCESS) {
            last_error_ = rc;
            throw std::runtime_error("CviVpssProcessor - CVI_VPSS_SetChnCrop failed");
        }
    }

    VIDEO_FRAME_INFO_S input_frame{};
    VB_BLK input_block = VB_INVALID_HANDLE;
    bool input_from_mat = false;

    if (frame.storage_type() == Frame::StorageType::CVI && frame.video_frame()) {
        input_frame = *frame.video_frame();
    } else {
        input_frame = mat_to_video_frame(frame.to_mat(), input_block, select_output_pool(frame.width(), frame.height(), input_format));
        input_from_mat = true;
    }

    CVI_S32 rc = CVI_SUCCESS;
    const int send_attempts = 3;
    const int send_timeout_ms = 200;
    for (int attempt = 0; attempt < send_attempts; ++attempt) {
        rc = CVI_VPSS_SendFrame(grp_, &input_frame, send_timeout_ms);
        if (rc == CVI_SUCCESS) {
            break;
        }
        if (rc != CVI_ERR_VPSS_NOBUF && rc != CVI_ERR_VPSS_BUF_EMPTY) {
            break;
        }
        usleep(5000);
    }
    if (input_from_mat && input_block != VB_INVALID_HANDLE) {
        CVI_VB_ReleaseBlock(input_block);
    }
    if (rc != CVI_SUCCESS) {
        if (use_crop) {
            crop_info.bEnable = CVI_FALSE;
            CVI_VPSS_SetChnCrop(grp_, chn_, &crop_info);
        }
        last_error_ = rc;
        std::ostringstream oss;
        oss << "CviVpssProcessor - CVI_VPSS_SendFrame failed: 0x" << std::hex << rc
            << " grp=" << grp_ << " chn=" << chn_
            << " input=" << input_frame.stVFrame.u32Width << "x" << input_frame.stVFrame.u32Height
            << " out=" << out_width << "x" << out_height;
        throw std::runtime_error(oss.str());
    }
    VIDEO_FRAME_INFO_S output_frame{};
    const int get_attempts = 5;
    for (int attempt = 0; attempt < get_attempts; ++attempt) {
        rc = CVI_VPSS_GetChnFrame(grp_, chn_, &output_frame, 100);
        if (rc == CVI_SUCCESS) {
            break;
        }
        if (rc != CVI_ERR_VPSS_NOBUF && rc != CVI_ERR_VPSS_BUF_EMPTY) {
            break;
        }
        usleep(5000);
    }

    if (use_crop) {
        crop_info.bEnable = CVI_FALSE;
        CVI_VPSS_SetChnCrop(grp_, chn_, &crop_info);
    }

    if (rc != CVI_SUCCESS) {
        last_error_ = rc;
        std::ostringstream oss;
        oss << "CviVpssProcessor - CVI_VPSS_GetChnFrame failed: 0x" << std::hex << rc;
        throw std::runtime_error(oss.str());
    }

    Frame out(output_frame, grp_, chn_);
    out.set_vpss_owner(grp_, chn_);
    frame = std::move(out);
}

VB_POOL CviVpssProcessor::select_output_pool(uint32_t width, uint32_t height, PixelFormat format) const {
    // find_pool handles RGB↔BGR fallback internally, so pass the exact format.
    if (MmfContext::instance().is_initialized()) {
        VB_POOL pool = MmfContext::instance().vb_plan().find_pool(width, height, format,
                                                                  VbPoolUsage::VPSS_PREPROCESS);
        if (pool != VB_INVALID_POOLID) {
            return pool;
        }
    }

    uint32_t needed = COMMON_GetPicBufferSize(width, height,
                                              to_cvi_pixel_format(format),
                                              DATA_BITWIDTH_8, COMPRESS_MODE_NONE, 0);
    if (needed == 0) {
        return VB_INVALID_POOLID;
    }

    VB_CONFIG_S vb_config{};
    CVI_S32 rc = CVI_VB_GetConfig(&vb_config);
    if (rc != CVI_SUCCESS || vb_config.u32MaxPoolCnt == 0) {
        return VB_INVALID_POOLID;
    }

    VB_POOL best_pool = VB_INVALID_POOLID;
    uint32_t best_size = 0;
    for (uint32_t i = 0; i < vb_config.u32MaxPoolCnt; ++i) {
        uint32_t blk_size = vb_config.astCommPool[i].u32BlkSize;
        uint32_t blk_cnt = vb_config.astCommPool[i].u32BlkCnt;
        if (blk_size == 0 || blk_cnt == 0) {
            continue;
        }
        if (blk_size >= needed && (best_pool == VB_INVALID_POOLID || blk_size < best_size)) {
            best_pool = static_cast<VB_POOL>(i);
            best_size = blk_size;
        }
    }

    return best_pool;
}

void CviVpssProcessor::destroy_group() {
    if (!group_created_) {
        return;
    }

    if (group_started_) {
        CVI_VPSS_DisableChn(grp_, chn_);
        CVI_VPSS_StopGrp(grp_);
        MmfContext::instance().set_mem_vpss_group_started(false);
        group_started_ = false;
    }

    if (out_pool_ != VB_INVALID_POOLID) {
        CVI_VPSS_DetachVbPool(grp_, chn_);
        out_pool_ = VB_INVALID_POOLID;
    }

    MmfContext::instance().release_mem_vpss_group();
    group_created_ = false;
    grp_ = -1;
    max_width_ = 0;
    max_height_ = 0;
    input_format_ = PixelFormat::UNKNOWN;
    out_width_ = 0;
    out_height_ = 0;
    out_format_ = PixelFormat::UNKNOWN;
    letterbox_enabled_ = false;
    pad_value_ = 0;
}
#endif

} // namespace lua_cv
