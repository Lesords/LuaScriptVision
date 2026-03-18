#include "model_lua_runtime.h"

#include "error_codes.h"
#include "modules/lua_cv.h"
#include "modules/lua_nn.h"
#include "modules/lua_utils.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include <cstdlib>

namespace node {

void configure_lua_path_from_script(const std::string& script_path) {
    const char* existing_lua_path = std::getenv("LUA_PATH");
    if (existing_lua_path) {
        return;
    }

    size_t last_slash = script_path.find_last_of("/\\");
    if (last_slash == std::string::npos) {
        return;
    }

    std::string script_dir = script_path.substr(0, last_slash);
    std::string parent_dir;
    size_t parent_slash = script_dir.find_last_of("/\\");
    if (parent_slash != std::string::npos) {
        parent_dir = script_dir.substr(0, parent_slash);
    }

    std::string lua_path;
    if (!parent_dir.empty()) {
        lua_path = parent_dir + "/?.lua;" + parent_dir + "/?/init.lua;";
    }
    lua_path += script_dir + "/?.lua;" + script_dir + "/?/init.lua;;";
    setenv("LUA_PATH", lua_path.c_str(), 1);
}

LuaRuntimeInitResult create_model_lua_runtime() {
    LuaRuntimeInitResult result;
    result.code = MA_OK;
    result.state = luaL_newstate();
    if (!result.state) {
        result.code = MA_ENOMEM;
        result.error_message = "Failed to create Lua state";
        return result;
    }

    luaL_openlibs(result.state);
    lua_cv::register_module(result.state);
    lua_nn::register_module(result.state);
    lua_utils::register_module(result.state);
    return result;
}

ModelScriptLoadResult load_model_script_bindings(lua_State* L, const std::string& script_path) {
    ModelScriptLoadResult result;
    result.code = MA_OK;

    if (luaL_dofile(L, script_path.c_str()) != LUA_OK) {
        std::string err = lua_tostring(L, -1);
        result.code = MA_EINVAL;
        result.error_message = "Lua script error: " + err;
        return result;
    }

    LuaIntf::LuaRef model = LuaIntf::LuaRef::popFromStack(L);
    if (!model.isTable()) {
        result.code = MA_EINVAL;
        result.error_message = "Script must return a table";
        model = LuaIntf::LuaRef();
        return result;
    }

    result.bindings.postprocess = model["postprocess"];
    result.bindings.select_rois = model["select_rois"];
    result.bindings.preprocess_config_ref = model["preprocess_config"];

    if (!result.bindings.postprocess.isFunction()) {
        result.code = MA_EINVAL;
        result.error_message = "Missing postprocess function in script";
        model = LuaIntf::LuaRef();
        return result;
    }

    if (result.bindings.preprocess_config_ref.isTable()) {
        result.bindings.preprocess_config =
            PreprocessConfig::fromLuaRef(result.bindings.preprocess_config_ref);
        result.bindings.preprocess_config_explicit = true;
    }

    model = LuaIntf::LuaRef();
    return result;
}

}  // namespace node
