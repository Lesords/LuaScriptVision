#include "frame.h"

#include <cstring>
#include <stdexcept>

#ifdef USE_CVI_MPI
#include <cvi_sys.h>
#include <cvi_vdec.h>
#endif

namespace lua_cv {

Frame::Frame() = default;

Frame::Frame(const cv::Mat& mat) : mat_(mat) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = PixelFormat::BGR;
    }
}

Frame::Frame(const cv::Mat& mat, PixelFormat format) : mat_(mat) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = format;
    }
}

Frame::Frame(cv::Mat&& mat) : mat_(std::move(mat)) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = PixelFormat::BGR;
    }
}

Frame::Frame(cv::Mat&& mat, PixelFormat format) : mat_(std::move(mat)) {
    if (!mat_.empty()) {
        storage_type_ = StorageType::OPENCV;
        format_ = format;
    }
}

#ifdef USE_CVI_MPI
Frame::Frame(const VIDEO_FRAME_INFO_S& frame, bool owns_memory)
    : cvi_frame_(frame) {
    storage_type_ = StorageType::CVI;
    format_ = from_cvi_pixel_format(frame.stVFrame.enPixelFormat);
    if (owns_memory) {
        owner_ = FrameOwner::VB;
        vb_block_ = CVI_VB_PhysAddr2Handle(frame.stVFrame.u64PhyAddr[0]);
    } else {
        owner_ = FrameOwner::EXTERNAL;
    }
}

Frame::Frame(const VIDEO_FRAME_INFO_S& frame, int vpss_grp, int vpss_chn)
    : cvi_frame_(frame) {
    storage_type_ = StorageType::CVI;
    format_ = from_cvi_pixel_format(frame.stVFrame.enPixelFormat);
    vpss_grp_ = vpss_grp;
    vpss_chn_ = vpss_chn;
    owner_ = FrameOwner::EXTERNAL;  // Caller must explicitly call set_vpss_owner() to take ownership
}
#endif

Frame::Frame(const Frame& other) {
    *this = other.clone();
}

Frame::Frame(Frame&& other) noexcept {
    *this = std::move(other);
}

Frame& Frame::operator=(const Frame& other) {
    if (this != &other) {
        *this = other.clone();
    }
    return *this;
}

Frame& Frame::operator=(Frame&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();

    storage_type_ = other.storage_type_;
    format_ = other.format_;
    mat_ = std::move(other.mat_);
    mat_cache_ = std::move(other.mat_cache_);
    mat_cache_valid_ = other.mat_cache_valid_;
    mapped_view_ = std::move(other.mapped_view_);
    mapped_ptr_ = other.mapped_ptr_;
    mapped_size_ = other.mapped_size_;

#ifdef USE_CVI_MPI
    cvi_frame_ = other.cvi_frame_;
    owner_ = other.owner_;
    vpss_grp_ = other.vpss_grp_;
    vpss_chn_ = other.vpss_chn_;
    vdec_chn_ = other.vdec_chn_;
    vb_block_ = other.vb_block_;

    // Transfer ownership - reset other's state
    other.owner_ = FrameOwner::EXTERNAL;
    other.vpss_grp_ = -1;
    other.vpss_chn_ = -1;
    other.vdec_chn_ = -1;
    other.vb_block_ = VB_INVALID_HANDLE;
#endif

    // Reset other's common state
    other.storage_type_ = StorageType::NONE;
    other.format_ = PixelFormat::UNKNOWN;
    other.mat_cache_valid_ = false;
    other.mapped_ptr_ = nullptr;
    other.mapped_size_ = 0;

    return *this;
}

Frame::~Frame() {
    release();
}

bool Frame::empty() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.empty();
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return cvi_frame_.stVFrame.u32Width == 0 || cvi_frame_.stVFrame.u32Height == 0;
    }
#endif
    return true;
}

int Frame::width() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.cols;
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return static_cast<int>(cvi_frame_.stVFrame.u32Width);
    }
#endif
    return 0;
}

int Frame::height() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.rows;
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return static_cast<int>(cvi_frame_.stVFrame.u32Height);
    }
#endif
    return 0;
}

int Frame::channels() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.channels();
    }
    return channels_for_format(format_);
}

bool Frame::has_physical_addr() const {
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return cvi_frame_.stVFrame.u64PhyAddr[0] != 0;
    }
#endif
    return false;
}

uint64_t Frame::physical_addr() const {
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        return static_cast<uint64_t>(cvi_frame_.stVFrame.u64PhyAddr[0]);
    }
#endif
    return 0;
}

const cv::Mat& Frame::to_mat_view() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_;
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        ensure_mapped();
        return mapped_view_;
    }
#endif
    // Return empty mat for NONE storage type
    return mat_;
}

cv::Mat Frame::to_mat_copy() const {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_.clone();
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        ensure_mapped();
        if (format_ == PixelFormat::NV12 || format_ == PixelFormat::NV21) {
            // Convert YUV to BGR for copy
            cv::Mat bgr;
            int code = (format_ == PixelFormat::NV21) ? cv::COLOR_YUV2BGR_NV21
                                                      : cv::COLOR_YUV2BGR_NV12;
            cv::cvtColor(mapped_view_, bgr, code);
            return bgr;
        }
        if (format_ == PixelFormat::RGB_PLANAR) {
            // Convert planar R,G,B to interleaved BGR
            int w = static_cast<int>(cvi_frame_.stVFrame.u32Width);
            int h = static_cast<int>(cvi_frame_.stVFrame.u32Height);
            int stride = static_cast<int>(cvi_frame_.stVFrame.u32Stride[0]);
            uint8_t* base = static_cast<uint8_t*>(mapped_ptr_);
            cv::Mat r_plane(h, w, CV_8UC1, base, stride);
            cv::Mat g_plane(h, w, CV_8UC1, base + stride * h, stride);
            cv::Mat b_plane(h, w, CV_8UC1, base + stride * h * 2, stride);
            cv::Mat bgr;
            std::vector<cv::Mat> channels = {b_plane, g_plane, r_plane};
            cv::merge(channels, bgr);
            return bgr;
        }
        if (format_ == PixelFormat::RGB) {
            // Convert interleaved RGB to BGR (OpenCV expects BGR)
            cv::Mat bgr;
            cv::cvtColor(mapped_view_, bgr, cv::COLOR_RGB2BGR);
            return bgr;
        }
        return mapped_view_.clone();
    }
#endif
    return cv::Mat();
}

const cv::Mat& Frame::to_mat() const {
    return const_cast<Frame*>(this)->to_mat();
}

cv::Mat& Frame::to_mat() {
    if (storage_type_ == StorageType::OPENCV) {
        return mat_;
    }
#ifdef USE_CVI_MPI
    if (storage_type_ == StorageType::CVI) {
        if (mat_cache_valid_) {
            return mat_cache_;
        }
        mat_cache_ = to_mat_copy();
        mat_cache_valid_ = true;
        return mat_cache_;
    }
#endif
    return mat_;
}

Frame Frame::clone() const {
    if (storage_type_ == StorageType::OPENCV) {
        return Frame(mat_.clone());
    }
    if (storage_type_ == StorageType::NONE) {
        return Frame();
    }
    cv::Mat cloned = to_mat().clone();
    return Frame(std::move(cloned));
}

void Frame::release() {
#ifdef USE_CVI_MPI
    switch (owner_) {
        case FrameOwner::VDEC:
            if (vdec_chn_ >= 0) {
                CVI_VDEC_ReleaseFrame(static_cast<VDEC_CHN>(vdec_chn_), &cvi_frame_);
            }
            break;
        case FrameOwner::VPSS:
            if (vpss_grp_ >= 0 && vpss_chn_ >= 0) {
                CVI_VPSS_ReleaseChnFrame(static_cast<VPSS_GRP>(vpss_grp_),
                                         static_cast<VPSS_CHN>(vpss_chn_),
                                         &cvi_frame_);
            }
            break;
        case FrameOwner::VB:
            if (vb_block_ != VB_INVALID_HANDLE) {
                CVI_VB_ReleaseBlock(vb_block_);
            }
            break;
        case FrameOwner::EXTERNAL:
            // Do not release - caller owns the resource
            break;
    }
    owner_ = FrameOwner::EXTERNAL;
#endif

    reset();
}

#ifdef USE_CVI_MPI
const VIDEO_FRAME_INFO_S* Frame::video_frame() const {
    if (storage_type_ != StorageType::CVI) {
        return nullptr;
    }
    return &cvi_frame_;
}

VIDEO_FRAME_INFO_S* Frame::video_frame() {
    if (storage_type_ != StorageType::CVI) {
        return nullptr;
    }
    return &cvi_frame_;
}

void Frame::set_vpss_owner(int vpss_grp, int vpss_chn) {
    vpss_grp_ = vpss_grp;
    vpss_chn_ = vpss_chn;
    owner_ = FrameOwner::VPSS;
    vb_block_ = VB_INVALID_HANDLE;
    vdec_chn_ = -1;
}

void Frame::set_vb_owner(VB_BLK vb_block) {
    vb_block_ = vb_block;
    owner_ = FrameOwner::VB;
    vpss_grp_ = -1;
    vpss_chn_ = -1;
    vdec_chn_ = -1;
}

void Frame::set_vdec_owner(int vdec_chn) {
    vdec_chn_ = vdec_chn;
    owner_ = FrameOwner::VDEC;
    vpss_grp_ = -1;
    vpss_chn_ = -1;
    vb_block_ = VB_INVALID_HANDLE;
}

std::shared_ptr<tensor::VbMemory> Frame::as_vb_memory() const {
    if (storage_type_ != StorageType::CVI) {
        return nullptr;
    }
    if (cvi_frame_.stVFrame.u64PhyAddr[0] == 0) {
        return nullptr;
    }

    // Calculate total frame size
    size_t total_size = static_cast<size_t>(cvi_frame_.stVFrame.u32Length[0]) +
                        static_cast<size_t>(cvi_frame_.stVFrame.u32Length[1]) +
                        static_cast<size_t>(cvi_frame_.stVFrame.u32Length[2]);
    if (total_size == 0) {
        // Fallback calculation for planar formats
        int h = static_cast<int>(cvi_frame_.stVFrame.u32Height);
        int stride = static_cast<int>(cvi_frame_.stVFrame.u32Stride[0]);
        if (format_ == PixelFormat::NV12 || format_ == PixelFormat::NV21) {
            total_size = static_cast<size_t>(stride) * static_cast<size_t>(h + h / 2);
        } else if (format_ == PixelFormat::RGB_PLANAR) {
            total_size = static_cast<size_t>(stride) * static_cast<size_t>(h) * 3;
        } else {
            total_size = static_cast<size_t>(stride) * static_cast<size_t>(h);
        }
    }
    VB_BLK block = vb_block_;
    if (block == VB_INVALID_HANDLE) {
        block = CVI_VB_PhysAddr2Handle(cvi_frame_.stVFrame.u64PhyAddr[0]);
        if (block == VB_INVALID_HANDLE) {
            return nullptr;
        }
    }

    // Create VbMemory wrapper - does NOT take ownership (Frame still owns the resource)
    return tensor::VbMemory::from_block(block, total_size, /*cached=*/true, /*take_ownership=*/false);
}

void Frame::ensure_mapped() const {
    if (mapped_ptr_ != nullptr) {
        return;  // Already mapped
    }
    if (storage_type_ != StorageType::CVI) {
        return;
    }
    if (cvi_frame_.stVFrame.u64PhyAddr[0] == 0) {
        throw std::runtime_error("Frame::ensure_mapped - CVI frame missing physical address");
    }

    size_t total_size = static_cast<size_t>(cvi_frame_.stVFrame.u32Length[0]) +
                        static_cast<size_t>(cvi_frame_.stVFrame.u32Length[1]) +
                        static_cast<size_t>(cvi_frame_.stVFrame.u32Length[2]);
    if (total_size == 0) {
        // Fallback calculation for planar formats
        int h = static_cast<int>(cvi_frame_.stVFrame.u32Height);
        int stride = static_cast<int>(cvi_frame_.stVFrame.u32Stride[0]);
        if (format_ == PixelFormat::NV12 || format_ == PixelFormat::NV21) {
            total_size = static_cast<size_t>(stride) * static_cast<size_t>(h + h / 2);
        } else if (format_ == PixelFormat::RGB_PLANAR) {
            total_size = static_cast<size_t>(stride) * static_cast<size_t>(h) * 3;
        } else {
            total_size = static_cast<size_t>(stride) * static_cast<size_t>(h);
        }
    }

    void* mapped = CVI_SYS_MmapCache(cvi_frame_.stVFrame.u64PhyAddr[0],
                                     static_cast<CVI_U32>(total_size));
    if (!mapped) {
        throw std::runtime_error("Frame::ensure_mapped - CVI_SYS_MmapCache failed");
    }
    CVI_SYS_IonInvalidateCache(cvi_frame_.stVFrame.u64PhyAddr[0], mapped,
                               static_cast<CVI_U32>(total_size));

    mapped_ptr_ = mapped;
    mapped_size_ = total_size;

    // Create view Mat based on format
    uint8_t* base = static_cast<uint8_t*>(mapped);
    int w = static_cast<int>(cvi_frame_.stVFrame.u32Width);
    int h = static_cast<int>(cvi_frame_.stVFrame.u32Height);
    int stride = static_cast<int>(cvi_frame_.stVFrame.u32Stride[0]);

    if (format_ == PixelFormat::BGR || format_ == PixelFormat::RGB) {
        mapped_view_ = cv::Mat(h, w, CV_8UC3, base, stride);
    } else if (format_ == PixelFormat::GRAY) {
        mapped_view_ = cv::Mat(h, w, CV_8UC1, base, stride);
    } else if (format_ == PixelFormat::NV12 || format_ == PixelFormat::NV21) {
        // Return raw YUV data as single-channel (no conversion for view)
        mapped_view_ = cv::Mat(h + h / 2, w, CV_8UC1, base, stride);
    } else if (format_ == PixelFormat::RGB_PLANAR) {
        // Planar R,G,B: expose as single-channel (3*h, w) for raw access
        mapped_view_ = cv::Mat(h * 3, w, CV_8UC1, base, stride);
    } else {
        unmap();
        throw std::runtime_error("Frame::ensure_mapped - unsupported CVI pixel format");
    }
}

void Frame::unmap() const {
    if (mapped_ptr_ != nullptr) {
        CVI_SYS_Munmap(mapped_ptr_, static_cast<CVI_U32>(mapped_size_));
        mapped_ptr_ = nullptr;
        mapped_size_ = 0;
        mapped_view_.release();
    }
}
#endif

void Frame::reset() {
    mat_.release();
    mat_cache_.release();
    mat_cache_valid_ = false;
#ifdef USE_CVI_MPI
    unmap();  // Unmap before resetting CVI frame
    std::memset(&cvi_frame_, 0, sizeof(cvi_frame_));
    owner_ = FrameOwner::EXTERNAL;
    vpss_grp_ = -1;
    vpss_chn_ = -1;
    vdec_chn_ = -1;
    vb_block_ = VB_INVALID_HANDLE;
#endif
    storage_type_ = StorageType::NONE;
    format_ = PixelFormat::UNKNOWN;
}

} // namespace lua_cv
