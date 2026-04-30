#pragma once

#include "LuaIntf.h"
#include <algorithm>
#include <nlohmann/json.hpp>

namespace node {

// Check if a Lua table is an array (consecutive integer keys starting from 1)
inline bool is_lua_array(const LuaIntf::LuaRef& ref) {
    if (!ref.isTable()) return false;

    int max_index = 0;
    int count = 0;
    bool has_string_key = false;

    for (auto& kv : ref) {
        LuaIntf::LuaRef key = kv.key<LuaIntf::LuaRef>();
        if (key.type() == LuaIntf::LuaTypeID::NUMBER) {
            int idx = key.toValue<int>();
            if (idx > 0) {
                max_index = std::max(max_index, idx);
                count++;
            } else {
                return false;  // Non-positive index (0 or negative)
            }
        } else if (key.type() == LuaIntf::LuaTypeID::STRING) {
            has_string_key = true;
        }
    }

    // Pure array: has integer keys, no string keys, and indices are consecutive
    // Empty table: treat as array (model postprocess returns empty array for no results)
    if (has_string_key) return false;
    if (count == 0) return true;   // Empty table -> array
    return count == max_index;     // Check consecutive: {1,2,3} count=3, max=3
}

// Convert LuaRef to JSON
inline nlohmann::json luaref_to_json(const LuaIntf::LuaRef& ref) {
    LuaIntf::LuaTypeID type = ref.type();
    if (type == LuaIntf::LuaTypeID::NONE || type == LuaIntf::LuaTypeID::NIL) return nullptr;
    if (type == LuaIntf::LuaTypeID::NUMBER) return ref.toValue<double>();
    if (type == LuaIntf::LuaTypeID::STRING) return ref.toValue<std::string>();
    if (type == LuaIntf::LuaTypeID::BOOLEAN) return ref.toValue<bool>();

    if (type == LuaIntf::LuaTypeID::TABLE) {
        if (is_lua_array(ref)) {
            // Array: consecutive integer keys starting from 1
            nlohmann::json arr = nlohmann::json::array();
            int len = ref.len();
            for (int i = 1; i <= len; ++i) {
                LuaIntf::LuaRef elem = ref[i].value<LuaIntf::LuaRef>();
                arr.push_back(luaref_to_json(elem));
            }
            return arr;
        }

        // Object (has string keys, or empty table, or sparse array)
        nlohmann::json obj = nlohmann::json::object();
        for (auto& kv : ref) {
            LuaIntf::LuaRef key_ref = kv.key<LuaIntf::LuaRef>();
            std::string key;

            if (key_ref.type() == LuaIntf::LuaTypeID::STRING) {
                key = key_ref.toValue<std::string>();
            } else if (key_ref.type() == LuaIntf::LuaTypeID::NUMBER) {
                // Non-consecutive integer key -> convert to string
                key = std::to_string(key_ref.toValue<int>());
            } else {
                continue;  // Skip unsupported key types
            }

            LuaIntf::LuaRef val = kv.value<LuaIntf::LuaRef>();
            obj[key] = luaref_to_json(val);
        }
        return obj;
    }

    return nullptr;  // Unsupported types (function, userdata, etc.)
}

} // namespace node
