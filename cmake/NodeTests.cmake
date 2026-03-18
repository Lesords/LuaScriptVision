# NodeTests.cmake - Phase 2 Node Framework test definitions
#
# This module defines all Node Framework test executables

if(NOT SG200X_BUILD)
    return()
endif()

message(STATUS "Configuring Phase 2 Node Framework tests...")

set(NODE_TEST_CAMERA_INCLUDES "")
if(ENABLE_CVI_CAMERA)
    list(APPEND NODE_TEST_CAMERA_INCLUDES ${CVI_ISP_INCLUDE_DIR})
endif()

# =============================================================================
# Test: Node Base Class
# =============================================================================
add_executable(test_node_base
    tests/node/test_node_base.cpp
    src/node/node.cpp
)

target_include_directories(test_node_base PRIVATE
    ${SRC_COMMON_INCLUDES}
    ${NLOHMANN_JSON_INCLUDE}
)

target_link_gtest(test_node_base WITH_MAIN)
target_link_libraries(test_node_base PRIVATE pthread)

message(STATUS "test_node_base: Node base class tests enabled")

# =============================================================================
# Test: Node Factory (with Camera+ISP support)
# =============================================================================
add_executable(test_node_factory
    tests/node/test_node_factory.cpp
    src/node/node.cpp
    src/node/node_factory.cpp
    src/node/node_server.cpp
    src/node/resource_estimator.cpp
    src/node/camera_node.cpp
    src/modules/cv/cvi_camera.cpp
    src/modules/cv/cvi_sensor.cpp
    ${CV_SOURCES}
    ${MEMORY_SOURCES}
    ${STREAM_SOURCES}
)

target_include_directories(test_node_factory PRIVATE
    ${SRC_COMMON_INCLUDES}
    ${NLOHMANN_JSON_INCLUDE}
    ${MOSQUITTO_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NODE_TEST_CAMERA_INCLUDES}
    ${CVI_RTSP_INCLUDE_DIR}
)

target_compile_definitions(test_node_factory PRIVATE USE_CVI_MPI USE_CVI_CAMERA)

target_link_directories(test_node_factory PRIVATE ${MOSQUITTO_LIB_DIR})

target_link_cvi_tpu(test_node_factory)
target_link_gtest(test_node_factory WITH_MAIN)
target_link_libraries(test_node_factory PRIVATE
    mosquitto
    ssl
    crypto
    cares
    ${OpenCV_LIBS}
)
target_link_cvi_isp(test_node_factory)
target_link_cvi_mpi(test_node_factory)
target_link_cvi_rtsp(test_node_factory)
target_link_system_libs(test_node_factory)

set_target_properties(test_node_factory PROPERTIES
    BUILD_RPATH "${MOSQUITTO_LIB_DIR};/mnt/system/usr/lib"
    INSTALL_RPATH "/usr/lib"
)

message(STATUS "test_node_factory: Node factory tests enabled (with Camera+ISP+MQTT)")

# =============================================================================
# Test: Executor
# =============================================================================
add_executable(test_executor
    tests/node/test_executor.cpp
    src/node/executor.cpp
)

target_include_directories(test_executor PRIVATE ${SRC_COMMON_INCLUDES})
target_link_gtest(test_executor WITH_MAIN)
target_link_libraries(test_executor PRIVATE pthread)

message(STATUS "test_executor: Executor tests enabled")

# =============================================================================
# Test: MessageBox
# =============================================================================
add_executable(test_message_box
    tests/node/test_message_box.cpp
    src/modules/cv/frame.cpp
    src/modules/cv/cv_types.cpp
)

target_include_directories(test_message_box PRIVATE
    ${SRC_COMMON_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
    ${CVI_BASE_INCLUDE_DIRS}
    ${NLOHMANN_JSON_INCLUDE}
)
target_link_gtest(test_message_box WITH_MAIN)
target_link_libraries(test_message_box PRIVATE
    pthread
    ${OpenCV_LIBS}
    m
    dl
    atomic
)
target_link_cvi_mpi(test_message_box)

message(STATUS "test_message_box: MessageBox and SharedFrame tests enabled")

# =============================================================================
# Test: MQTT Protocol (with Camera+ISP support)
# =============================================================================
message(STATUS "test_mqtt_protocol: ENABLED (mosquitto + OpenSSL + c-ares + Camera+ISP)")

add_executable(test_mqtt_protocol
    tests/node/test_mqtt_protocol.cpp
    src/node/node.cpp
    src/node/node_factory.cpp
    src/node/node_server.cpp
    src/node/executor.cpp
    src/node/resource_estimator.cpp
    src/node/camera_node.cpp
    src/modules/cv/cvi_camera.cpp
    src/modules/cv/cvi_sensor.cpp
    ${CV_SOURCES}
    ${MEMORY_SOURCES}
    ${STREAM_SOURCES}
)

target_include_directories(test_mqtt_protocol PRIVATE
    ${SRC_COMMON_INCLUDES}
    ${NLOHMANN_JSON_INCLUDE}
    ${MOSQUITTO_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NODE_TEST_CAMERA_INCLUDES}
    ${CVI_RTSP_INCLUDE_DIR}
)

target_compile_definitions(test_mqtt_protocol PRIVATE USE_CVI_MPI USE_CVI_CAMERA)

target_link_directories(test_mqtt_protocol PRIVATE ${MOSQUITTO_LIB_DIR})

target_link_cvi_tpu(test_mqtt_protocol)
target_link_gtest(test_mqtt_protocol WITH_MAIN)
target_link_libraries(test_mqtt_protocol PRIVATE
    mosquitto
    ssl
    crypto
    cares
    ${OpenCV_LIBS}
)
target_link_cvi_isp(test_mqtt_protocol)
target_link_cvi_mpi(test_mqtt_protocol)
target_link_cvi_rtsp(test_mqtt_protocol)
target_link_system_libs(test_mqtt_protocol)

set_target_properties(test_mqtt_protocol PROPERTIES
    BUILD_RPATH "${MOSQUITTO_LIB_DIR};/mnt/system/usr/lib"
    INSTALL_RPATH "/usr/lib"
)

# =============================================================================
# Test: ModelNode (requires TPU and Lua)
# =============================================================================
add_executable(test_model_node
    tests/node/test_model_node.cpp
    src/node/node.cpp
    src/node/node_factory.cpp
    src/node/node_server.cpp
    src/node/camera_node.cpp
    src/node/model_node.cpp
    src/node/model_preprocess_utils.cpp
    src/node/model_roi_utils.cpp
    src/node/executor.cpp
    src/node/resource_estimator.cpp
    ${TENSOR_SOURCES}
    ${MEMORY_SOURCES}
    ${CV_SOURCES}
    ${MODULE_SOURCES}
    ${BINDING_SOURCES}
    ${INFERENCE_SOURCES}
    ${STREAM_SOURCES}
)

target_include_directories(test_model_node PRIVATE
    ${SRC_COMMON_INCLUDES}
    ${LUAINTF_INCLUDES}
    ${NLOHMANN_JSON_INCLUDE}
    ${MOSQUITTO_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NODE_TEST_CAMERA_INCLUDES}
    ${CVI_RTSP_INCLUDE_DIR}
)

target_link_libraries(test_model_node PRIVATE
    -Wl,--whole-archive
    lua
    -Wl,--no-whole-archive
)
target_link_cvi_tpu(test_model_node)
target_link_gtest(test_model_node WITH_MAIN)
target_link_cvi_mpi(test_model_node)
if(ENABLE_CVI_CAMERA)
    target_link_cvi_isp(test_model_node)
endif()
target_link_directories(test_model_node PRIVATE ${MOSQUITTO_LIB_DIR})
target_link_libraries(test_model_node PRIVATE
    mosquitto
    ssl
    crypto
    cares
    ${OpenCV_LIBS}
)
target_link_cvi_rtsp(test_model_node)
target_link_system_libs(test_model_node)

message(STATUS "test_model_node: ModelNode tests enabled")

# =============================================================================
# Test: Error Codes
# =============================================================================
add_executable(test_error_codes tests/node/test_error_codes.cpp)

target_include_directories(test_error_codes PRIVATE ${SRC_COMMON_INCLUDES})
target_link_gtest(test_error_codes WITH_MAIN)
target_link_libraries(test_error_codes PRIVATE pthread)

message(STATUS "test_error_codes: Error codes tests enabled")

# =============================================================================
# Test: Resource Estimator (with Camera+ISP support)
# =============================================================================
add_executable(test_resource_estimator
    tests/node/test_resource_estimator.cpp
    src/node/resource_estimator.cpp
    src/node/node.cpp
    src/node/node_factory.cpp
    src/node/node_server.cpp
    src/node/camera_node.cpp
    src/modules/cv/cvi_camera.cpp
    src/modules/cv/cvi_sensor.cpp
    ${CV_SOURCES}
    ${MEMORY_SOURCES}
    ${STREAM_SOURCES}
)

target_include_directories(test_resource_estimator PRIVATE
    ${SRC_COMMON_INCLUDES}
    ${NLOHMANN_JSON_INCLUDE}
    ${MOSQUITTO_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${NODE_TEST_CAMERA_INCLUDES}
    ${CVI_RTSP_INCLUDE_DIR}
)

target_compile_definitions(test_resource_estimator PRIVATE USE_CVI_MPI USE_CVI_CAMERA)

target_link_directories(test_resource_estimator PRIVATE ${MOSQUITTO_LIB_DIR})

target_link_cvi_tpu(test_resource_estimator)
target_link_gtest(test_resource_estimator WITH_MAIN)
target_link_libraries(test_resource_estimator PRIVATE
    mosquitto
    ssl
    crypto
    cares
    ${OpenCV_LIBS}
)
target_link_cvi_isp(test_resource_estimator)
target_link_cvi_mpi(test_resource_estimator)
target_link_cvi_rtsp(test_resource_estimator)
target_link_system_libs(test_resource_estimator)

set_target_properties(test_resource_estimator PROPERTIES
    BUILD_RPATH "${MOSQUITTO_LIB_DIR};/mnt/system/usr/lib"
    INSTALL_RPATH "/usr/lib"
)

message(STATUS "test_resource_estimator: Resource estimator tests enabled (with Camera+ISP)")

message(STATUS "Phase 2 Node Framework tests configured")
