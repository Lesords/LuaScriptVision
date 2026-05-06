#pragma once

#include "LuaIntf.h"
#include "model_roi_utils.h"

#include <nlohmann/json.hpp>

struct lua_State;

namespace node {

std::vector<SelectedRoi> select_valid_rois(lua_State* L,
                                           const LuaIntf::LuaRef& selector,
                                           int frame_width,
                                           int frame_height,
                                           const nlohmann::json& upstream);

}  // namespace node
