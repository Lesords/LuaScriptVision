#pragma once

#include "LuaIntf.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct lua_State;

namespace node {

struct LuaModelPostprocessResult {
    nlohmann::json value = nlohmann::json::object();
    bool has_warning = false;
    nlohmann::json warning_payload;
    bool has_error = false;
    std::string error_message;
};

LuaModelPostprocessResult call_lua_model_postprocess(
    lua_State* L,
    const LuaIntf::LuaRef& postprocess,
    const std::vector<std::string>* output_names,
    std::vector<std::vector<float>> outputs,
    std::vector<std::vector<int64_t>> output_shapes,
    const nlohmann::json& meta);

}  // namespace node
