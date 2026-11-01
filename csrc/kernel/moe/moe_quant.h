// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Quantization recipe of the MoE operator: one method per weight group.  The
// public recipe strings are the ones halops.moe speaks; this header is the
// single place that maps them onto the host dispatch and the supported set.

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "kernel/moe/moe_configs.cuh"

namespace halops::moe {

struct MoERecipe {
    QuantMethod experts;
    QuantMethod latent_down;
    QuantMethod shared;
};

constexpr bool operator==(const MoERecipe& lhs, const MoERecipe& rhs) {
    return lhs.experts == rhs.experts && lhs.latent_down == rhs.latent_down &&
           lhs.shared == rhs.shared;
}

inline constexpr MoERecipe kBlockwiseRecipe{QuantMethod::kBlockwiseFP8,
                                            QuantMethod::kBlockwiseFP8,
                                            QuantMethod::kBlockwiseFP8};

// Recipes this build implements.  Extended one kernel milestone at a time.
inline constexpr MoERecipe kLatentMXFP8Recipe{QuantMethod::kBlockwiseFP8,
                                              QuantMethod::kMXFP8,
                                              QuantMethod::kBlockwiseFP8};

inline constexpr std::array<MoERecipe, 2> kSupportedRecipes{kBlockwiseRecipe,
                                                            kLatentMXFP8Recipe};

inline bool ParseQuantMethod(std::string_view name, QuantMethod& method) {
    if (name == "w8a8-blockwise") {
        method = QuantMethod::kBlockwiseFP8;
        return true;
    }
    if (name == "w8a8-mxfp8") {
        method = QuantMethod::kMXFP8;
        return true;
    }
    if (name == "w4a8-mxfp4") {
        method = QuantMethod::kMXFP4;
        return true;
    }
    return false;
}

inline const char* QuantMethodName(QuantMethod method) {
    switch (method) {
        case QuantMethod::kBlockwiseFP8:
            return "w8a8-blockwise";
        case QuantMethod::kMXFP8:
            return "w8a8-mxfp8";
        case QuantMethod::kMXFP4:
            return "w4a8-mxfp4";
        case QuantMethod::kNone:
            return "none";
    }
    return "?";
}

inline std::string RecipeToString(const MoERecipe& recipe) {
    return std::string("experts=") + QuantMethodName(recipe.experts) +
           ",latent_down=" + QuantMethodName(recipe.latent_down) +
           ",shared=" + QuantMethodName(recipe.shared);
}

inline std::string SupportedRecipesToString() {
    std::string out;
    for (const MoERecipe& recipe : kSupportedRecipes) {
        if (!out.empty()) {
            out += "; ";
        }
        out += RecipeToString(recipe);
    }
    return out;
}

inline bool IsSupportedRecipe(const MoERecipe& recipe) {
    for (const MoERecipe& supported : kSupportedRecipes) {
        if (recipe == supported) {
            return true;
        }
    }
    return false;
}

// Parses the wire form "experts=<method>,latent_down=<method>,shared=<method>".
inline bool ParseRecipe(std::string_view encoded, MoERecipe& recipe,
                        std::string& error) {
    bool seen[3] = {false, false, false};
    std::size_t pos = 0;
    while (pos <= encoded.size()) {
        const std::size_t comma = encoded.find(',', pos);
        const std::string_view item = encoded.substr(
            pos, comma == std::string_view::npos ? std::string_view::npos
                                                 : comma - pos);
        const std::size_t eq = item.find('=');
        if (eq == std::string_view::npos) {
            error = "expected group=recipe, got '" + std::string(item) + "'";
            return false;
        }
        const std::string_view group = item.substr(0, eq);
        const std::string_view name = item.substr(eq + 1);
        QuantMethod method{};
        if (!ParseQuantMethod(name, method)) {
            error = "unknown recipe '" + std::string(name) + "'";
            return false;
        }
        int index = -1;
        if (group == "experts") {
            index = 0;
            recipe.experts = method;
        } else if (group == "latent_down") {
            index = 1;
            recipe.latent_down = method;
        } else if (group == "shared") {
            index = 2;
            recipe.shared = method;
        } else {
            error = "unknown group '" + std::string(group) + "'";
            return false;
        }
        if (seen[index]) {
            error = "duplicate group '" + std::string(group) + "'";
            return false;
        }
        seen[index] = true;
        if (comma == std::string_view::npos) {
            break;
        }
        pos = comma + 1;
    }
    if (!(seen[0] && seen[1] && seen[2])) {
        error = "expected all of experts, latent_down, shared";
        return false;
    }
    return true;
}

}  // namespace halops::moe
