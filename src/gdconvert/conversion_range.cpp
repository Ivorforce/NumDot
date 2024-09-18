#include "conversion_range.h"

#include <cstddef>                        // for ptrdiff_t
#include <cstdint>                        // for int64_t
#include <stdexcept>                      // for runtime_error
#include "godot_cpp/variant/variant.hpp"  // for Variant
#include "ndrange.h"                      // for range_part
#include "xtensor/xslice.hpp"             // for xtuph

range_part to_range_part(const Variant& variant) {
    switch (variant.get_type()) {
        case Variant::INT:
            return static_cast<std::ptrdiff_t>(static_cast<int64_t>(variant));
        case Variant::NIL:
            return xt::placeholders::xtuph{};
        default:
            throw std::runtime_error("Invalid type for range.");
    }
}
