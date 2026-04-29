#pragma once

#ifdef USE_CVI_TPU

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "inference_session.h"
#include <cviruntime.h>
#include <cviruntime_context.h>
#include <linux/cvi_comm_video.h>

// Include actual headers for memory types
#include "memory/tpu_memory.h"
#if defined(USE_CVI_MPI)
#include "memory/vb_memory.h"
#endif

namespace inference {

class CviSession : public InferenceSession {
public:
    struct RunStats {
        double input_ms = 0.0;
        double forward_ms = 0.0;
        double output_ms = 0.0;
    };

    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        CVI_FMT fmt = CVI_FMT_FP32;
        float qscale = 0.0f;
        int zero_point = 0;
        CVI_NN_PIXEL_FORMAT_E pixel_format = CVI_NN_PIXEL_TENSOR;
        std::array<float, 3> mean{};
        std::array<float, 3> scale{};
    };

    struct VbInputSpec {
        PIXEL_FORMAT_E pixel_format = PIXEL_FORMAT_MAX;
        uint32_t width = 0;
        uint32_t height = 0;
        bool normalized = false;
        std::array<float, 3> mean{};
        std::array<float, 3> scale{};
    };

    // model_path only: self-managed RT handle (standalone/test use)
    explicit CviSession(const std::string& model_path);
    // Shared RT handle: used by SessionManager (no CVI_RT_Init/DeInit)
    CviSession(const std::string& model_path, CVI_RT_HANDLE shared_rt_handle);
    ~CviSession() override;

    std::pair<std::vector<float>, std::vector<int64_t>>
    run(const float* input_data, const std::vector<int64_t>& input_shape) override;

    void run_all(const float* input_data,
                 const std::vector<int64_t>& input_shape,
                 std::vector<std::vector<float>>* outputs,
                 std::vector<std::vector<int64_t>>* output_shapes);

    void run_all_selected(const float* input_data,
                          const std::vector<int64_t>& input_shape,
                          const std::vector<int32_t>& output_indices,
                          std::vector<std::vector<float>>* outputs,
                          std::vector<std::vector<int64_t>>* output_shapes);

    void run_vb(uint64_t input_phys_addr,
                size_t input_size,
                std::vector<std::vector<float>>* outputs,
                std::vector<std::vector<int64_t>>* output_shapes);
    void run_vb_selected(uint64_t input_phys_addr,
                         size_t input_size,
                         const std::vector<int32_t>& output_indices,
                         std::vector<std::vector<float>>* outputs,
                         std::vector<std::vector<int64_t>>* output_shapes);
#if defined(USE_CVI_MPI)
    void run_vb(const std::shared_ptr<tensor::VbMemory>& input,
                std::vector<std::vector<float>>* outputs,
                std::vector<std::vector<int64_t>>* output_shapes);
    void run_vb_selected(const std::shared_ptr<tensor::VbMemory>& input,
                         const std::vector<int32_t>& output_indices,
                         std::vector<std::vector<float>>* outputs,
                         std::vector<std::vector<int64_t>>* output_shapes);
#endif

    bool supports_vb_input() const;
    VbInputSpec get_vb_input_spec() const;

    std::vector<int64_t> get_input_shape(size_t index = 0) const override;
    std::vector<int64_t> get_output_shape(size_t index = 0) const override;

    const std::vector<std::string>& get_input_names() const override { return input_names_; }
    const std::vector<std::string>& get_output_names() const override { return output_names_; }

    TensorInfo get_input_info(size_t index = 0) const;
    TensorInfo get_output_info(size_t index = 0) const;

    int32_t input_count() const { return input_num_; }
    int32_t output_count() const { return output_num_; }
    int32_t primary_output_index() const { return primary_output_index_; }
    const RunStats& last_run_stats() const { return last_run_stats_; }

    const char* backend_name() const override { return "cviruntime"; }

private:
    int32_t select_primary_output_index() const;
    void init_model(const std::string& model_path);
    void init_device_buffers();
    void prepare_input(const float* input_data,
                       const std::vector<int64_t>& input_shape,
                       std::vector<int64_t>* actual_shape,
                       std::vector<float>* padded_input,
                       const float** input_ptr) const;
    void copy_input_to_device(const float* input_ptr, int64_t input_elements);
    std::vector<float> read_output_tensor(int32_t output_index) const;
    void collect_selected_outputs(const std::vector<int32_t>& output_indices,
                                  std::vector<std::vector<float>>* outputs,
                                  std::vector<std::vector<int64_t>>* output_shapes) const;

    CVI_RT_HANDLE rt_handle_ = nullptr;
    bool owns_rt_handle_ = false;  // true only for standalone constructor
    CVI_MODEL_HANDLE model_ = nullptr;
    CVI_TENSOR* input_tensors_ = nullptr;
    CVI_TENSOR* output_tensors_ = nullptr;
    int32_t input_num_ = 0;
    int32_t output_num_ = 0;
    int32_t primary_output_index_ = 0;
    RunStats last_run_stats_{};

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::shared_ptr<tensor::CviTpuMemory>> input_buffers_;
    std::vector<int32_t> all_output_indices_;
};

} // namespace inference

#endif  // USE_CVI_TPU
