// Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
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

#ifndef ISSUE_URL_H
#define ISSUE_URL_H

#include <string>
#include <vector>

namespace jax_metallib {

inline std::string UnsupportedOpsMessage(const std::vector<std::string>& op_names) {

    std::string op_list;
    for (size_t i = 0; i < op_names.size(); i++) {
        if (i > 0)
            op_list += ", ";
        op_list += op_names[i];
    }

    std::string encoded = op_names[0];
    for (size_t pos = 0; (pos = encoded.find('.', pos)) != std::string::npos; pos += 3) {
        encoded.replace(pos, 1, "%2E");
    }

    return "Unsupported operation(s): " + op_list +
           ". The MPS backend does not have a handler for these operations.\n\n"
           "File an issue: "
           "https://github.com/erfanzar/jax-metallib/issues/new?template=missing-op.yml"
           "&title=Missing+op:+" +
           encoded + "&op-name=" + encoded +
           "\n"
           "Add it yourself: https://github.com/erfanzar/jax-metallib/blob/main/CONTRIBUTING.md";
}

} // namespace jax_metallib

#endif
