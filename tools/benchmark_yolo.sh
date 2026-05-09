#!/bin/bash
# YOLO Performance Benchmark Script
# Usage: ./scripts/benchmark_yolo.sh [iterations]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/models"
IMAGE_DIR="$PROJECT_DIR/images"

# Default iterations
ITERATIONS=${1:-3}

# Check if build exists
if [ ! -f "$BUILD_DIR/lua_runner" ]; then
    echo "Error: lua_runner not found. Please build the project first."
    echo "  mkdir build && cd build && cmake .. && make"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "  YOLO Performance Benchmark"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Date: $(date)"
echo ""

# Test configurations: script,model,description
declare -a TESTS=(
    "yolo11_tensor_detector.lua,yolo11n.onnx,YOLO11n Detection (Tensor API)"
    "yolo11_tensor_pose.lua,yolo11n-pose.onnx,YOLO11n Pose (Tensor API)"
    "yolo11_tensor_seg.lua,yolo11n-seg.onnx,YOLO11n Segmentation (Tensor API)"
    "yolov5_tensor_detector.lua,yolov5n.onnx,YOLOv5n Detection (Tensor API)"
    "yolo11_detector.lua,yolo11n.onnx,YOLO11n Detection (Legacy)"
    "yolo11_pose.lua,yolo11n-pose.onnx,YOLO11n Pose (Legacy)"
    "yolo11_seg.lua,yolo11n-seg.onnx,YOLO11n Segmentation (Legacy)"
    "yolov5_detector.lua,yolov5n.onnx,YOLOv5n Detection (Legacy)"
)

# Results storage (in-memory array instead of file)
declare -a RESULTS=()

# Function to measure execution time and save full output for accuracy validation
measure_time_and_save_output() {
    local cmd="$1"
    local output_file="$2"

    local start_ns=$(date +%s%N)
    eval "$cmd" > "$output_file" 2>&1
    local exit_code=$?
    local end_ns=$(date +%s%N)

    if [ $exit_code -ne 0 ]; then
        echo "-1"
        return 1
    fi

    # Calculate time in milliseconds
    local elapsed_ns=$((end_ns - start_ns))
    local elapsed_ms=$((elapsed_ns / 1000000))
    echo "$elapsed_ms"
}

# Function to extract detection info from output
extract_detection_info() {
    local output_file="$1"
    local info_file="$2"

    # Extract detection count
    local detection_count=0
    if grep -q "Total:.*detections" "$output_file"; then
        detection_count=$(grep "Total:.*detections" "$output_file" | grep -oP '\d+(?= detections)')
    elif grep -q "NMS后最终框:" "$output_file"; then
        detection_count=$(grep "NMS后最终框:" "$output_file" | grep -oP '(?<=: )\d+')
    fi

    # Extract bounding boxes and labels
    # Pattern: "Box N: label (x, y, w, h) conf=score"
    grep -E "^Box [0-9]+:" "$output_file" | sort > "$info_file"

    echo "$detection_count"
}

# Run tests
echo "Running benchmarks..."
echo ""

for test_config in "${TESTS[@]}"; do
    IFS=',' read -r script model description <<< "$test_config"

    script_path="$SCRIPT_DIR/$script"
    model_path="$MODEL_DIR/$model"
    image_path="$IMAGE_DIR/zidane.jpg"

    # Check if files exist
    if [ ! -f "$script_path" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $description - Script not found: $script"
        continue
    fi
    if [ ! -f "$model_path" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $description - Model not found: $model"
        continue
    fi

    echo -e "${BLUE}[TEST]${NC} $description"

    # Warmup run (not counted)
    "$BUILD_DIR/lua_runner" "$script_path" "$model_path" "$image_path" > /dev/null 2>&1

    total_time=0
    success=true
    times=()
    detections=()
    cmd="\"$BUILD_DIR/lua_runner\" \"$script_path\" \"$model_path\" \"$image_path\""

    # Save first run output for accuracy validation
    first_output_file="$PROJECT_DIR/.bench_output_$$_$(echo "$script" | sed 's/[^a-zA-Z0-9]/_/g')"
    first_info_file="${first_output_file}.info"

    for i in $(seq 1 $ITERATIONS); do
        if [ $i -eq 1 ]; then
            # First run: save full output
            time_ms=$(measure_time_and_save_output "$cmd" "$first_output_file")
            det_count=$(extract_detection_info "$first_output_file" "$first_info_file")
        else
            # Subsequent runs: discard output
            output_tmp=$(mktemp)
            time_ms=$(measure_time_and_save_output "$cmd" "$output_tmp")
            det_count=$(extract_detection_info "$output_tmp" "/dev/null")
            rm -f "$output_tmp"
        fi

        if [ "$time_ms" = "-1" ]; then
            echo -e "${RED}[FAIL]${NC} Iteration $i failed"
            success=false
            break
        fi

        times+=("$time_ms")
        detections+=("$det_count")
        total_time=$((total_time + time_ms))
    done

    if [ "$success" = true ] && [ ${#times[@]} -gt 0 ]; then
        avg_time=$((total_time / ${#times[@]}))
        min_time=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
        max_time=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)

        # Get first detection count (should be consistent across runs)
        det_count="${detections[0]}"

        # Check if detection count is consistent
        det_consistent=true
        for d in "${detections[@]}"; do
            if [ "$d" != "$det_count" ]; then
                det_consistent=false
                break
            fi
        done

        if [ "$det_consistent" = true ]; then
            echo -e "${GREEN}[PASS]${NC} Avg: ${avg_time}ms, Min: ${min_time}ms, Max: ${max_time}ms | Detections: ${det_count}"
            RESULTS+=("$description: avg=${avg_time}ms min=${min_time}ms max=${max_time}ms detections=${det_count}")
        else
            echo -e "${YELLOW}[WARN]${NC} Avg: ${avg_time}ms | Detections: inconsistent ${detections[*]}"
            RESULTS+=("$description: avg=${avg_time}ms detections=INCONSISTENT")
        fi
    else
        RESULTS+=("$description: FAILED")
    fi
    echo ""
done

echo "========================================"
echo ""

# Also run C++ baseline if available
if [ -f "$BUILD_DIR/cpp_infer" ]; then
    echo -e "${BLUE}[TEST]${NC} C++ Baseline (Pure C++, YOLOv5n only)"

    # Warmup
    "$BUILD_DIR/cpp_infer" "$MODEL_DIR/yolov5n.onnx" "$IMAGE_DIR/zidane.jpg" > /dev/null 2>&1

    total_time=0
    times=()
    detections=()
    cmd="\"$BUILD_DIR/cpp_infer\" \"$MODEL_DIR/yolov5n.onnx\" \"$IMAGE_DIR/zidane.jpg\""

    # Save first run output
    first_output_file="$PROJECT_DIR/.bench_output_$$_cpp_infer"
    first_info_file="${first_output_file}.info"

    for i in $(seq 1 $ITERATIONS); do
        if [ $i -eq 1 ]; then
            time_ms=$(measure_time_and_save_output "$cmd" "$first_output_file")
            det_count=$(extract_detection_info "$first_output_file" "$first_info_file")
        else
            output_tmp=$(mktemp)
            time_ms=$(measure_time_and_save_output "$cmd" "$output_tmp")
            det_count=$(extract_detection_info "$output_tmp" "/dev/null")
            rm -f "$output_tmp"
        fi

        if [ "$time_ms" != "-1" ]; then
            times+=("$time_ms")
            detections+=("$det_count")
            total_time=$((total_time + time_ms))
        fi
    done

    if [ ${#times[@]} -gt 0 ]; then
        avg_time=$((total_time / ${#times[@]}))
        min_time=$(printf '%s\n' "${times[@]}" | sort -n | head -1)
        max_time=$(printf '%s\n' "${times[@]}" | sort -n | tail -1)
        det_count="${detections[0]}"

        echo -e "${GREEN}[PASS]${NC} Avg: ${avg_time}ms, Min: ${min_time}ms, Max: ${max_time}ms | Detections: ${det_count}"
        RESULTS+=("C++ Baseline (YOLOv5n): avg=${avg_time}ms min=${min_time}ms max=${max_time}ms detections=${det_count}")
    fi
    echo ""
fi

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo "Benchmark Results - $(date)"
echo "Iterations: $ITERATIONS"
echo "========================================"
for result in "${RESULTS[@]}"; do
    echo "$result"
done

echo ""
echo "========================================"
echo "  Accuracy Validation"
echo "========================================"

# Helper function to compare detection outputs
compare_detections() {
    local file1="$1"
    local file2="$2"
    local name1="$3"
    local name2="$4"

    if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
        echo -e "${YELLOW}⊘${NC} Cannot compare $name1 vs $name2 - files missing"
        return 1
    fi

    local count1=$(wc -l < "$file1")
    local count2=$(wc -l < "$file2")

    if [ "$count1" != "$count2" ]; then
        echo -e "${RED}✗${NC} $name1 vs $name2: Different counts ($count1 vs $count2)"
        return 1
    fi

    # Compare detection boxes using diff
    local diff_output=$(diff "$file1" "$file2" 2>/dev/null)
    if [ -z "$diff_output" ]; then
        echo -e "${GREEN}✓${NC} $name1 vs $name2: Identical ($count1 detections)"
        return 0
    else
        # Show first few differences
        echo -e "${YELLOW}△${NC} $name1 vs $name2: Same count but different boxes"
        echo "  First difference:"
        echo "$diff_output" | head -6 | sed 's/^/    /'
        return 1
    fi
}

# Extract detection counts for quick comparison
echo "Detection Counts:"
yolo11n_tensor=$(printf '%s\n' "${RESULTS[@]}" | grep "YOLO11n Detection (Tensor API)" | grep -oP 'detections=\K\d+' || echo "N/A")
yolo11n_legacy=$(printf '%s\n' "${RESULTS[@]}" | grep "YOLO11n Detection (Legacy)" | grep -oP 'detections=\K\d+' || echo "N/A")
yolov5n_tensor=$(printf '%s\n' "${RESULTS[@]}" | grep "YOLOv5n Detection (Tensor API)" | grep -oP 'detections=\K\d+' || echo "N/A")
yolov5n_legacy=$(printf '%s\n' "${RESULTS[@]}" | grep "YOLOv5n Detection (Legacy)" | grep -oP 'detections=\K\d+' || echo "N/A")
cpp_baseline=$(printf '%s\n' "${RESULTS[@]}" | grep "C++ Baseline" | grep -oP 'detections=\K\d+' || echo "N/A")

echo "  YOLO11n Tensor: $yolo11n_tensor | Legacy: $yolo11n_legacy"
echo "  YOLOv5n Tensor: $yolov5n_tensor | Legacy: $yolov5n_legacy | C++: $cpp_baseline"
echo ""

# Detailed comparison using saved detection info
echo "Detailed Box Comparison:"
all_consistent=true

# YOLO11n: Tensor vs Legacy
yolo11n_tensor_info="$PROJECT_DIR/.bench_output_$$_yolo11_tensor_detector_lua.info"
yolo11n_legacy_info="$PROJECT_DIR/.bench_output_$$_yolo11_detector_lua.info"
if ! compare_detections "$yolo11n_tensor_info" "$yolo11n_legacy_info" "YOLO11n Tensor" "YOLO11n Legacy"; then
    all_consistent=false
fi

# YOLOv5n: Tensor vs Legacy
yolov5n_tensor_info="$PROJECT_DIR/.bench_output_$$_yolov5_tensor_detector_lua.info"
yolov5n_legacy_info="$PROJECT_DIR/.bench_output_$$_yolov5_detector_lua.info"
if ! compare_detections "$yolov5n_tensor_info" "$yolov5n_legacy_info" "YOLOv5n Tensor" "YOLOv5n Legacy"; then
    all_consistent=false
fi

# YOLOv5n: Tensor vs C++
cpp_info="$PROJECT_DIR/.bench_output_$$_cpp_infer.info"
if ! compare_detections "$yolov5n_tensor_info" "$cpp_info" "YOLOv5n Tensor" "C++ Baseline"; then
    all_consistent=false
fi

echo ""
if [ "$all_consistent" = true ]; then
    echo -e "${GREEN}Overall: All implementations produce identical detection results${NC}"
else
    echo -e "${YELLOW}Overall: Minor differences detected - verify thresholds and NMS params${NC}"
fi

# Cleanup temporary files
echo ""
echo "Cleaning up temporary files..."
rm -f "$PROJECT_DIR"/.bench_output_$$_* 2>/dev/null || true
