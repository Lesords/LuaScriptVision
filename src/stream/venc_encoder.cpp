#include "venc_encoder.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace lua_cv {

#ifdef USE_CVI_MPI

namespace {
bool trace_blocking_enabled() {
    static int enabled = -1;
    if (enabled >= 0) {
        return enabled == 1;
    }
    const char* value = std::getenv("LSV_TRACE_BLOCKING");
    enabled = (value && value[0] != '\0' && value[0] != '0') ? 1 : 0;
    return enabled == 1;
}

std::atomic<uint64_t> g_getstream_seq{0};
}  // namespace

VencEncoder::VencEncoder(const Config& config)
    : config_(config) {
}

VencEncoder::~VencEncoder() {
    shutdown();
}

bool VencEncoder::init() {
    if (initialized_) {
        return true;
    }

    bool success = false;
    switch (config_.codec) {
        case CodecType::H264:
            success = init_h264();
            break;
        case CodecType::H265:
            success = init_h265();
            break;
        case CodecType::JPEG:
            success = init_jpeg();
            break;
        case CodecType::MJPEG:
            success = init_mjpeg();
            break;
    }

    if (!success) {
        std::cerr << "[VENC] Failed to initialize encoder, channel=" << config_.channel << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "[VENC] Encoder initialized: channel=" << config_.channel
              << " " << config_.width << "x" << config_.height << std::endl;
    return true;
}

void VencEncoder::shutdown() {
    if (!initialized_) {
        return;
    }

    if (stream_acquired_) {
        release_stream();
    }

    if (bound_) {
        unbind_from_vpss();
    }

    CVI_S32 rc = CVI_SUCCESS;

    if (receiving_) {
        rc = CVI_VENC_StopRecvFrame(config_.channel);
        if (rc != CVI_SUCCESS) {
            std::cerr << "[VENC] CVI_VENC_StopRecvFrame failed: rc=" << rc
                      << " channel=" << config_.channel << std::endl;
        }
        receiving_ = false;
    }

    rc = CVI_VENC_ResetChn(config_.channel);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_ResetChn failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
    }

    rc = CVI_VENC_DestroyChn(config_.channel);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_DestroyChn failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
    }

    initialized_ = false;
    std::cout << "[VENC] Encoder shutdown: channel=" << config_.channel << std::endl;
}

bool VencEncoder::start_recv() {
    VENC_RECV_PIC_PARAM_S recv_param;
    recv_param.s32RecvPicNum = -1;

    CVI_S32 rc = CVI_VENC_StartRecvFrame(config_.channel, &recv_param);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_StartRecvFrame failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        return false;
    }
    receiving_ = true;
    return true;
}

bool VencEncoder::init_h264() {
    VENC_CHN_ATTR_S chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));

    chn_attr.stVencAttr.enType = PT_H264;
    chn_attr.stVencAttr.u32MaxPicWidth = config_.width;
    chn_attr.stVencAttr.u32MaxPicHeight = config_.height;
    chn_attr.stVencAttr.u32PicWidth = config_.width;
    chn_attr.stVencAttr.u32PicHeight = config_.height;
    chn_attr.stVencAttr.u32BufSize = config_.width * config_.height * 3 / 2;
    chn_attr.stVencAttr.u32Profile = H264E_PROFILE_HIGH;
    chn_attr.stVencAttr.bByFrame = CVI_TRUE;
    chn_attr.stVencAttr.bSingleCore = CVI_FALSE;
    chn_attr.stVencAttr.bEsBufQueueEn = CVI_TRUE;
    chn_attr.stVencAttr.bIsoSendFrmEn = CVI_TRUE;

    chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
    chn_attr.stRcAttr.stH264Cbr.u32BitRate = config_.bitrate_kbps;
    chn_attr.stRcAttr.stH264Cbr.u32Gop = config_.gop;
    chn_attr.stRcAttr.stH264Cbr.u32SrcFrameRate = config_.fps;
    chn_attr.stRcAttr.stH264Cbr.fr32DstFrameRate = config_.fps;
    chn_attr.stRcAttr.stH264Cbr.u32StatTime = 2;
    chn_attr.stRcAttr.stH264Cbr.bVariFpsEn = CVI_FALSE;

    chn_attr.stGopAttr.enGopMode = VENC_GOPMODE_NORMALP;
    chn_attr.stGopAttr.stNormalP.s32IPQpDelta = config_.ip_qp_delta;

    CVI_S32 rc = CVI_VENC_CreateChn(config_.channel, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_CreateChn(H264) failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        return false;
    }

    if (!start_recv()) {
        CVI_VENC_DestroyChn(config_.channel);
        return false;
    }

    return true;
}

bool VencEncoder::init_h265() {
    VENC_CHN_ATTR_S chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));

    chn_attr.stVencAttr.enType = PT_H265;
    chn_attr.stVencAttr.u32MaxPicWidth = config_.width;
    chn_attr.stVencAttr.u32MaxPicHeight = config_.height;
    chn_attr.stVencAttr.u32PicWidth = config_.width;
    chn_attr.stVencAttr.u32PicHeight = config_.height;
    chn_attr.stVencAttr.u32BufSize = config_.width * config_.height * 3 / 2;
    chn_attr.stVencAttr.u32Profile = 0;
    chn_attr.stVencAttr.bByFrame = CVI_TRUE;
    chn_attr.stVencAttr.bSingleCore = CVI_FALSE;
    chn_attr.stVencAttr.bEsBufQueueEn = CVI_TRUE;
    chn_attr.stVencAttr.bIsoSendFrmEn = CVI_TRUE;

    chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H265CBR;
    chn_attr.stRcAttr.stH265Cbr.u32BitRate = config_.bitrate_kbps;
    chn_attr.stRcAttr.stH265Cbr.u32Gop = config_.gop;
    chn_attr.stRcAttr.stH265Cbr.u32SrcFrameRate = config_.fps;
    chn_attr.stRcAttr.stH265Cbr.fr32DstFrameRate = config_.fps;
    chn_attr.stRcAttr.stH265Cbr.u32StatTime = 2;
    chn_attr.stRcAttr.stH265Cbr.bVariFpsEn = CVI_FALSE;

    chn_attr.stGopAttr.enGopMode = VENC_GOPMODE_NORMALP;
    chn_attr.stGopAttr.stNormalP.s32IPQpDelta = config_.ip_qp_delta;

    CVI_S32 rc = CVI_VENC_CreateChn(config_.channel, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_CreateChn(H265) failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        return false;
    }

    if (!start_recv()) {
        CVI_VENC_DestroyChn(config_.channel);
        return false;
    }

    return true;
}

bool VencEncoder::init_jpeg() {
    VENC_CHN_ATTR_S chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));

    chn_attr.stVencAttr.enType = PT_JPEG;
    chn_attr.stVencAttr.u32MaxPicWidth = config_.width;
    chn_attr.stVencAttr.u32MaxPicHeight = config_.height;
    chn_attr.stVencAttr.u32PicWidth = config_.width;
    chn_attr.stVencAttr.u32PicHeight = config_.height;
    chn_attr.stVencAttr.u32BufSize = config_.width * config_.height * 3 / 2;
    chn_attr.stVencAttr.bByFrame = CVI_TRUE;
    chn_attr.stVencAttr.bSingleCore = CVI_FALSE;
    chn_attr.stVencAttr.bEsBufQueueEn = CVI_TRUE;
    chn_attr.stVencAttr.bIsoSendFrmEn = CVI_TRUE;

    CVI_S32 rc = CVI_VENC_CreateChn(config_.channel, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_CreateChn(JPEG) failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        return false;
    }

    VENC_JPEG_PARAM_S jpeg_param;
    std::memset(&jpeg_param, 0, sizeof(jpeg_param));
    jpeg_param.u32Qfactor = config_.quality;

    rc = CVI_VENC_SetJpegParam(config_.channel, &jpeg_param);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_SetJpegParam failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        CVI_VENC_DestroyChn(config_.channel);
        return false;
    }

    if (!start_recv()) {
        CVI_VENC_DestroyChn(config_.channel);
        return false;
    }

    return true;
}

bool VencEncoder::init_mjpeg() {
    VENC_CHN_ATTR_S chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));

    chn_attr.stVencAttr.enType = PT_MJPEG;
    chn_attr.stVencAttr.u32MaxPicWidth = config_.width;
    chn_attr.stVencAttr.u32MaxPicHeight = config_.height;
    chn_attr.stVencAttr.u32PicWidth = config_.width;
    chn_attr.stVencAttr.u32PicHeight = config_.height;
    // JPEG encoder requires bitstream buffer size aligned to 1024 bytes.
    chn_attr.stVencAttr.u32BufSize = ((config_.width * config_.height * 3 / 2) + 1023U) & ~1023U;
    chn_attr.stVencAttr.bByFrame = CVI_TRUE;
    chn_attr.stVencAttr.bSingleCore = CVI_FALSE;
    chn_attr.stVencAttr.bEsBufQueueEn = CVI_TRUE;
    chn_attr.stVencAttr.bIsoSendFrmEn = CVI_FALSE;  // FALSE for VPSS-bound continuous MJPEG

    chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_MJPEGCBR;
    chn_attr.stRcAttr.stMjpegCbr.u32BitRate = config_.bitrate_kbps;
    chn_attr.stRcAttr.stMjpegCbr.u32SrcFrameRate = config_.fps;
    chn_attr.stRcAttr.stMjpegCbr.fr32DstFrameRate = config_.fps;
    chn_attr.stRcAttr.stMjpegCbr.u32StatTime = 2;
    chn_attr.stRcAttr.stMjpegCbr.bVariFpsEn = CVI_FALSE;

    CVI_S32 rc = CVI_VENC_CreateChn(config_.channel, &chn_attr);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_CreateChn(MJPEG) failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        return false;
    }

    if (!start_recv()) {
        CVI_VENC_DestroyChn(config_.channel);
        return false;
    }

    return true;
}

bool VencEncoder::bind_to_vpss(VPSS_GRP grp, VPSS_CHN chn) {
    if (!initialized_) {
        std::cerr << "[VENC] Cannot bind: encoder not initialized" << std::endl;
        return false;
    }

    if (bound_) {
        std::cerr << "[VENC] Already bound to VPSS" << std::endl;
        return false;
    }

    MMF_CHN_S src_chn;
    src_chn.enModId = CVI_ID_VPSS;
    src_chn.s32DevId = grp;
    src_chn.s32ChnId = chn;

    MMF_CHN_S dst_chn;
    dst_chn.enModId = CVI_ID_VENC;
    dst_chn.s32DevId = 0;
    dst_chn.s32ChnId = config_.channel;

    bool was_receiving = receiving_;
    if (receiving_) {
        CVI_S32 stop_rc = CVI_VENC_StopRecvFrame(config_.channel);
        if (stop_rc != CVI_SUCCESS) {
            std::cerr << "[VENC] CVI_VENC_StopRecvFrame before bind failed: rc=" << stop_rc
                      << " channel=" << config_.channel << std::endl;
            return false;
        }
        receiving_ = false;
    }

    CVI_S32 rc = CVI_SYS_Bind(&src_chn, &dst_chn);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_SYS_Bind failed: rc=" << rc
                  << " VPSS[" << grp << "," << chn << "] -> VENC[" << config_.channel << "]"
                  << std::endl;
        if (was_receiving) {
            start_recv();
        }
        return false;
    }

    bound_ = true;
    bound_vpss_grp_ = grp;
    bound_vpss_chn_ = chn;

    if (!start_recv()) {
        CVI_SYS_UnBind(&src_chn, &dst_chn);
        bound_ = false;
        bound_vpss_grp_ = -1;
        bound_vpss_chn_ = -1;
        return false;
    }

    std::cout << "[VENC] Bound VPSS[" << grp << "," << chn << "] -> VENC[" << config_.channel << "]"
              << std::endl;
    return true;
}

void VencEncoder::unbind_from_vpss() {
    if (!bound_) {
        return;
    }

    MMF_CHN_S src_chn;
    src_chn.enModId = CVI_ID_VPSS;
    src_chn.s32DevId = bound_vpss_grp_;
    src_chn.s32ChnId = bound_vpss_chn_;

    MMF_CHN_S dst_chn;
    dst_chn.enModId = CVI_ID_VENC;
    dst_chn.s32DevId = 0;
    dst_chn.s32ChnId = config_.channel;

    CVI_S32 rc = CVI_SYS_UnBind(&src_chn, &dst_chn);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_SYS_UnBind failed: rc=" << rc << std::endl;
    }

    bound_ = false;
    bound_vpss_grp_ = -1;
    bound_vpss_chn_ = -1;

    std::cout << "[VENC] Unbound from VPSS" << std::endl;
}

bool VencEncoder::send_frame(const VIDEO_FRAME_INFO_S& frame, int timeout_ms) {
    if (!initialized_) {
        std::cerr << "[VENC] Cannot send frame: encoder not initialized" << std::endl;
        return false;
    }

    CVI_S32 rc = CVI_VENC_SendFrame(config_.channel, &frame, timeout_ms);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_SendFrame failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        return false;
    }

    return true;
}

bool VencEncoder::get_stream(EncodedStream* stream, int timeout_ms) {
    if (!initialized_) {
        std::cerr << "[VENC] Cannot get stream: encoder not initialized" << std::endl;
        return false;
    }

    if (!stream) {
        std::cerr << "[VENC] Invalid stream pointer" << std::endl;
        return false;
    }

    if (stream_acquired_) {
        release_stream();
    }

    VENC_CHN_STATUS_S status;
    CVI_S32 rc = CVI_VENC_QueryStatus(config_.channel, &status);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_QueryStatus failed: rc=" << rc << std::endl;
        return false;
    }

    if (status.u32CurPacks == 0) {
        return false;
    }

    std::memset(&current_stream_, 0, sizeof(current_stream_));

    current_packs_.resize(status.u32CurPacks);
    current_stream_.pstPack = current_packs_.data();
    current_stream_.u32PackCount = status.u32CurPacks;

    uint64_t trace_id = 0;
    auto trace_start = std::chrono::steady_clock::now();
    if (trace_blocking_enabled()) {
        trace_id = ++g_getstream_seq;
        std::cerr << "[VENC] GetStream begin id=" << trace_id
                  << " channel=" << config_.channel
                  << " packs=" << status.u32CurPacks
                  << " timeout_ms=" << timeout_ms << std::endl;
    }

    rc = CVI_VENC_GetStream(config_.channel, &current_stream_, timeout_ms);
    if (trace_blocking_enabled()) {
        auto trace_end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              trace_end - trace_start)
                              .count();
        std::cerr << "[VENC] GetStream end id=" << trace_id
                  << " rc=0x" << std::hex << rc << std::dec
                  << " elapsed_ms=" << elapsed_ms << std::endl;
    }

    if (rc != CVI_SUCCESS) {
        current_stream_.pstPack = nullptr;
        current_packs_.clear();
        if (rc != CVI_ERR_VENC_BUF_EMPTY &&
            rc != CVI_ERR_VENC_BUSY &&
            rc != CVI_ERR_VENC_EMPTY_STREAM_FRAME &&
            rc != CVI_ERR_VENC_EMPTY_PACK) {
            std::cerr << "[VENC] CVI_VENC_GetStream failed: rc=" << rc
                      << " channel=" << config_.channel << std::endl;
        }
        return false;
    }

    stream_acquired_ = true;

    stream->data.clear();
    stream->is_keyframe = false;
    stream->pts = 0;

    if (current_stream_.u32PackCount == 0 || current_stream_.pstPack == nullptr) {
        return true;
    }

    stream->pts = current_stream_.pstPack[0].u64PTS;

    for (uint32_t i = 0; i < current_stream_.u32PackCount; ++i) {
        VENC_PACK_S* pack = &current_stream_.pstPack[i];
        if (pack->pu8Addr == nullptr || pack->u32Len == 0) {
            continue;
        }

        uint8_t* data_ptr = pack->pu8Addr + pack->u32Offset;
        uint32_t data_len = pack->u32Len - pack->u32Offset;

        stream->data.insert(stream->data.end(), data_ptr, data_ptr + data_len);

        if (config_.codec == CodecType::H264) {
            if (pack->DataType.enH264EType == H264E_NALU_IDRSLICE ||
                pack->DataType.enH264EType == H264E_NALU_SPS) {
                stream->is_keyframe = true;
            }
        } else if (config_.codec == CodecType::H265) {
            if (pack->DataType.enH265EType == H265E_NALU_IDRSLICE ||
                pack->DataType.enH265EType == H265E_NALU_VPS ||
                pack->DataType.enH265EType == H265E_NALU_SPS) {
                stream->is_keyframe = true;
            }
        } else if (config_.codec == CodecType::JPEG || config_.codec == CodecType::MJPEG) {
            stream->is_keyframe = true;
        }
    }

    return true;
}

void VencEncoder::release_stream() {
    if (!stream_acquired_) {
        return;
    }

    CVI_S32 rc = CVI_VENC_ReleaseStream(config_.channel, &current_stream_);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_ReleaseStream failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
    }

    current_stream_.pstPack = nullptr;
    current_packs_.clear();
    stream_acquired_ = false;
}

bool VencEncoder::request_idr() {
    if (!initialized_) {
        return false;
    }

    if (config_.codec != CodecType::H264 && config_.codec != CodecType::H265) {
        return false;
    }

    CVI_S32 rc = CVI_VENC_RequestIDR(config_.channel, CVI_TRUE);
    if (rc != CVI_SUCCESS) {
        std::cerr << "[VENC] CVI_VENC_RequestIDR failed: rc=" << rc
                  << " channel=" << config_.channel << std::endl;
        return false;
    }

    return true;
}

#endif

} // namespace lua_cv
