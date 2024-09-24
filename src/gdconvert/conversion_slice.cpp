#include "conversion_slice.h"

#include <cstdint>                            // for int64_t
#include <stdexcept>                          // for runtime_error
#include <variant>                            // for visit
#include "godot_cpp/classes/object.hpp"       // for Object
#include "godot_cpp/core/object.hpp"          // for Object::cast_to
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "godot_cpp/variant/variant.hpp"      // for Variant
#include "ndrange.h"                          // for NDRange
#include "xtensor/xslice.hpp"                 // for xall_tag, xellipsis_tag


// TODO Somehow, I can't manage to make this a lambda function visitor.
struct ToRangeVisitor {
    template <typename T1, typename T2, typename T3>
    xt::xstrided_slice<std::ptrdiff_t> operator()(T1 a, T2 b, T3 c) {
        return xt::range(a, b, c);
    }
};

StringName newaxis() {
    const StringName newaxis = StringName("newaxis");
    return newaxis;
}

StringName ellipsis() {
    const StringName ellipsis = StringName("...");
    return ellipsis;
}

xt::xstrided_slice<std::ptrdiff_t> variant_to_slice_part(const Variant& variant)  {
    auto type = variant.get_type();

    switch (type) {
        case Variant::OBJECT:
            if (auto ndrange = Object::cast_to<NDRange>(variant)) {
                return std::visit(ToRangeVisitor{}, ndrange->start, ndrange->stop, ndrange->step);
            }
        break;
        case Variant::NIL:
            return xt::all();
        case Variant::INT:
            return static_cast<int64_t>(variant);
        case Variant::STRING_NAME:
            if (StringName(variant) == ::newaxis()) {
                return xt::newaxis();
            }
            else if (StringName(variant) == ::ellipsis()) {
                return xt::ellipsis();
            }
        break;
        default:
            break;
    }

    throw std::runtime_error("Variant cannot be converted to a slice.");
}

xt::xstrided_slice_vector variants_to_slice_vector(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
    xt::xstrided_slice_vector sv(arg_count);
    for (int i = 0; i < arg_count; i++) {
        sv[i] = variant_to_slice_part(*args[i]);
    }
    return sv;
}

range_part variant_to_range_part(const Variant& variant) {
    switch (variant.get_type()) {
        case Variant::INT:
            return static_cast<std::ptrdiff_t>(static_cast<int64_t>(variant));
        case Variant::NIL:
            return xt::placeholders::xtuph{};
        default:
            throw std::runtime_error("Invalid type for range.");
    }
}
