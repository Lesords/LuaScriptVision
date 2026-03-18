#pragma once

#include "LuaIntf.h"

#include <nlohmann/json.hpp>

struct lua_State;

namespace node {

LuaIntf::LuaRef json_to_luaref(lua_State* L, const nlohmann::json& value);

}  // namespace node
