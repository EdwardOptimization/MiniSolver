#pragma once

#include <type_traits>

#include "minisolver/core/solver_options.h"

namespace minisolver {
namespace detail {

// Optional marker emitted by MiniModel.py. Fused/generated paths are valid only
// for the integrator used at code-generation time.
template <typename, typename = void>
struct has_generated_integrator : std::false_type { };

template <typename Model>
struct has_generated_integrator<Model, std::void_t<decltype(Model::generated_integrator)>>
    : std::true_type { };

template <typename Model>
static constexpr bool has_generated_integrator_v = has_generated_integrator<Model>::value;

template <typename Model>
bool generated_integrator_matches(IntegratorType integrator)
{
    if constexpr (has_generated_integrator_v<Model>) {
        return Model::generated_integrator == integrator;
    } else {
        return true;
    }
}

} // namespace detail
} // namespace minisolver
