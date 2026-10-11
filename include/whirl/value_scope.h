// SPDX-License-Identifier: Apache-2.0
// ValueScope: set a variable for the lifetime of a scope and restore the previous value on every
// exit path, including a thrown exception. Used for state that must not leak into the next call
// (Model::attn_win, the MTP draft attention window).
#pragma once

#include <utility>

namespace whirl {

template <class T>
class ValueScope {
public:
    ValueScope(T& var, T value) : var_(var), saved_(std::move(var)) { var_ = std::move(value); }
    ~ValueScope() { var_ = std::move(saved_); }
    ValueScope(const ValueScope&) = delete;
    ValueScope& operator=(const ValueScope&) = delete;

private:
    T& var_;
    T saved_;
};

}  // namespace whirl
