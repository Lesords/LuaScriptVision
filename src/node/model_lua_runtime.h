#pragma once

#include "preprocess_config.h"

#include "LuaIntf.h"

#include <string>

struct lua_State;

namespace node {

struct LuaRuntimeInitResult {
    int code = 0;
    std::string error_message;
    lua_State* state = nullptr;
};

struct ModelScriptBindings {
    LuaIntf::LuaRef postprocess;
    LuaIntf::LuaRef select_rois;
    LuaIntf::LuaRef preprocess_config_ref;
    PreprocessConfig preprocess_config;
    bool preprocess_config_explicit = false;
};

struct ModelScriptLoadResult {
    int code = 0;
    std::string error_message;
    ModelScriptBindings bindings;
};

void configure_lua_path_from_script(const std::string& script_path);
LuaRuntimeInitResult create_model_lua_runtime();
ModelScriptLoadResult load_model_script_bindings(lua_State* L, const std::string& script_path);

}  // namespace node
