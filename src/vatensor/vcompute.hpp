#ifndef VCOMPUTE_INPLACE_H
#define VCOMPUTE_INPLACE_H

#include <utility>                      // for forward
#include <variant>                      // for visit, variant
#include <vector>                       // for vector
#include "varray.hpp"                     // for VArrayTarget, VScalar, VData
#include "vassign.hpp"                    // for assign_nonoverlapping, broadc...
#include "vpromote.hpp"                   // for promote_value_type_if_needed
#include "dtype.hpp"
#include "xtensor/xstorage.hpp"         // for uvector
#include "xtensor/xtensor_forward.hpp"  // for xarray

namespace va {
    template<typename OutputType, typename Result>
    std::shared_ptr<VArray> create_varray(VStoreAllocator& allocator, const Result& result) {
        using RNatural = typename std::decay_t<decltype(result)>::value_type;
        using OStorable = promote::compatible_type_or_64_bit_t<OutputType, VScalar>;

        static_assert(std::is_convertible_v<RNatural, OStorable>, "Cannot store the function result.");

        const auto dimension = result.dimension();

        // Create new array, assign to our target pointer.
        // OutputType may be different from R, if we want different behavior than xtensor for computation.
        std::shared_ptr<VStore> result_store = allocator.allocate(va::dtype_of_type<OStorable>(), result.size());
        auto data = make_compute<OStorable*>(
            static_cast<OStorable*>(result_store->data()),
            promote::promote_list_if_needed<shape_type>(result.shape()),
            strides_type{}, // unused
            dimension <= 1 ? xt::layout_type::any : xt::layout_type::row_major
        );

        va::broadcasting_assign_typesafe(data, result);

        return std::make_shared<VArray>(
            VArray {
                std::move(result_store),
                std::move(data),
                0
            }
        );
    }
}

#endif //VCOMPUTE_INPLACE_H
