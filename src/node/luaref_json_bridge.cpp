#include "luaref_json_bridge.h"

#include <lua.h>

namespace node {
namespace {

void push_json(lua_State* L, const nlohmann::json& value) {
    if (value.is_null()) {
        lua_pushnil(L);
    } else if (value.is_boolean()) {
        lua_pushboolean(L, value.get<bool>());
    } else if (value.is_number_integer()) {
        lua_pushinteger(L, static_cast<lua_Integer>(value.get<int64_t>()));
    } else if (value.is_number_unsigned()) {
        lua_pushinteger(L, static_cast<lua_Integer>(value.get<uint64_t>()));
    } else if (value.is_number_float()) {
        lua_pushnumber(L, static_cast<lua_Number>(value.get<double>()));
    } else if (value.is_string()) {
        lua_pushstring(L, value.get<std::string>().c_str());
    } else if (value.is_array()) {
        lua_newtable(L);
        int index = 1;
        for (const auto& item : value) {
            push_json(L, item);
            lua_rawseti(L, -2, index++);
        }
    } else if (value.is_object()) {
        lua_newtable(L);
        for (auto it = value.begin(); it != value.end(); ++it) {
            lua_pushstring(L, it.key().c_str());
            push_json(L, it.value());
            lua_settable(L, -3);
        }
    } else {
        lua_pushnil(L);
    }
}

}  // namespace

LuaIntf::LuaRef json_to_luaref(lua_State* L, const nlohmann::json& value) {
    push_json(L, value);
    return LuaIntf::LuaRef::popFromStack(L);
}

}  // namespace node
