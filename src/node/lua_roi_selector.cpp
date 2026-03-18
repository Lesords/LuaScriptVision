#include "lua_roi_selector.h"

#include "luaref_json.h"
#include "luaref_json_bridge.h"

namespace node {

std::vector<Roi> select_valid_rois(lua_State* L,
                                   const LuaIntf::LuaRef& selector,
                                   int frame_width,
                                   int frame_height,
                                   const nlohmann::json& upstream) {
    if (!selector.isFunction()) {
        return {};
    }

    LuaIntf::LuaRef upstream_ref = json_to_luaref(L, upstream);
    LuaIntf::LuaRef rois_ref = selector.call<LuaIntf::LuaRef>(upstream_ref);
    nlohmann::json rois_json = luaref_to_json(rois_ref);
    if (!rois_json.is_array() || rois_json.empty()) {
        return {};
    }

    std::vector<Roi> rois;
    rois.reserve(rois_json.size());
    for (const auto& roi_item : rois_json) {
        Roi roi;
        if (!parse_roi(roi_item, &roi)) {
            continue;
        }
        if (clamp_roi_to_bounds(frame_width, frame_height, &roi)) {
            rois.push_back(roi);
        }
    }
    return rois;
}

}  // namespace node
