#include "lua_model_postprocess.h"

#include "luaref_json.h"
#include "luaref_json_bridge.h"
#include "tensor/tensor.h"

#include <lua.h>

namespace node {

LuaModelPostprocessResult call_lua_model_postprocess(
    lua_State* L,
    const LuaIntf::LuaRef& postprocess,
    const std::vector<std::string>* output_names,
    std::vector<std::vector<float>> outputs,
    std::vector<std::vector<int64_t>> output_shapes,
    const nlohmann::json& meta) {
    LuaModelPostprocessResult bridge_result;

    try {
        if (outputs.size() != output_shapes.size()) {
            bridge_result.has_warning = true;
            bridge_result.warning_payload = {
                {"message", "Output count mismatch"},
                {"outputs", outputs.size()},
                {"shapes", output_shapes.size()}
            };
        }

        LuaIntf::LuaRef outputs_ref = LuaIntf::LuaRef::createTable(L);
        const size_t count = std::min(outputs.size(), output_shapes.size());
        for (size_t i = 0; i < count; ++i) {
            tensor::Tensor tensor(std::move(outputs[i]), output_shapes[i]);
            const std::string default_name = "output" + std::to_string(i);
            outputs_ref[default_name] = tensor;
            if (output_names && i < output_names->size()) {
                const std::string& name = (*output_names)[i];
                if (!name.empty() && name != default_name) {
                    outputs_ref[name] = tensor;
                }
            }
        }

        LuaIntf::LuaRef meta_ref = json_to_luaref(L, meta);
        LuaIntf::LuaRef result = postprocess.call<LuaIntf::LuaRef>(outputs_ref, meta_ref);
        bridge_result.value = luaref_to_json(result);

        result = LuaIntf::LuaRef();
        outputs_ref = LuaIntf::LuaRef();
        meta_ref = LuaIntf::LuaRef();
        if (L) {
            lua_gc(L, LUA_GCSTEP, 200);
        }
    } catch (const LuaIntf::LuaException& e) {
        bridge_result.has_error = true;
        bridge_result.error_message = "Postprocess Lua error: " + std::string(e.what());
        bridge_result.value = nlohmann::json::object();
    } catch (const std::exception& e) {
        bridge_result.has_error = true;
        bridge_result.error_message = "Postprocess error: " + std::string(e.what());
        bridge_result.value = nlohmann::json::object();
    }

    return bridge_result;
}

}  // namespace node
