#include "ndi.h"

#include <vatensor/reduce.h>                // for max, mean, min, prod, std
#include <cmath>                            // for int64_t
#include <cstddef>                          // for ptrdiff_t, size_t
#include <functional>                       // for function
#include <memory>                           // for make_shared
#include <optional>                         // for optional
#include <stdexcept>                        // for runtime_error
#include <type_traits>                      // for decay_t
#include <utility>                          // for move
#include <variant>                          // for visit, variant
#include <vector>                           // for vector
#include <vatensor/linalg.h>
#include <vatensor/vassign.h>
#include "gdconvert/conversion_array.h"     // for variant_as_array
#include "gdconvert/conversion_ints.h"     // for variant_as_shape
#include "gdconvert/conversion_slice.h"     // for ellipsis, newaxis
#include "godot_cpp/core/error_macros.hpp"  // for ERR_FAIL_V_MSG
#include "godot_cpp/core/memory.hpp"        // for _post_initialize, memnew
#include "ndarray.h"                        // for NDArray
#include "vatensor/varray.h"                // for VArrayTarget, DType, VArray
#include "xtensor/xslice.hpp"               // for xtuph
#include "xtensor/xtensor_forward.hpp"      // for xarray


using namespace godot;

void ndi::_bind_methods() {
	godot::ClassDB::bind_static_method("ndi", D_METHOD("sum", "a"), &ndi::sum);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("prod", "a"), &ndi::prod);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("max", "a"), &ndi::max);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("min", "a"), &ndi::min);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("norm", "a", "ord"), &ndi::norm, DEFVAL(nullptr), DEFVAL(2));

	godot::ClassDB::bind_static_method("ndi", D_METHOD("reduce_dot", "a", "b"), &ndi::reduce_dot);
}

ndi::ndi() = default;
ndi::~ndi() = default;

template <typename Visitor, typename... Args>
inline int64_t reduction(Visitor&& visitor, const Args&... args) {
	try {
		const auto result = std::forward<Visitor>(visitor)(variant_as_array(args)...);

		if constexpr (std::is_same_v<std::decay_t<decltype(result)>, va::VScalar>) {
			return va::scalar_to_type<int64_t>(result);
		}
		else {
			return result;
		}
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

#define REDUCTION1(func, varray1) \
	reduction([](const va::VArray& array) { return va::func(array); }, (varray1))

#define REDUCTION2(func, varray1, varray2) \
	reduction([](const va::VArray& x1, const va::VArray& x2) { return va::func(x1, x2); }, (varray1), (varray2))

int64_t ndi::sum(const Variant& a) {
	return REDUCTION1(sum, a);
}

int64_t ndi::prod(const Variant& a) {
	return REDUCTION1(prod, a);
}

int64_t ndi::max(const Variant& a) {
	return REDUCTION1(max, a);
}

int64_t ndi::min(const Variant& a) {
	return REDUCTION1(min, a);
}

int64_t ndi::norm(const Variant& a, const Variant& ord) {
	switch (ord.get_type()) {
		case Variant::INT:
			switch (static_cast<int64_t>(ord)) {
				case 0:
					return REDUCTION1(norm_l0, a);
				case 1:
					return REDUCTION1(norm_l1, a);
				case 2:
					return REDUCTION1(norm_l2, a);
				default:
					break;
			}
		case Variant::FLOAT:
			if (std::isinf(static_cast<int64_t>(ord))) {
				return REDUCTION1(norm_linf, a);
			}
		default:
			break;
	}

	ERR_FAIL_V_MSG({}, "This norm is currently not supported");
}

//int64_t ndi::all(const Variant& a) {
//    return REDUCTION1(all, a);
//}
//
//int64_t ndi::any(const Variant& a) {
//    return REDUCTION1(any, a);
//}

int64_t ndi::reduce_dot(const Variant& a, const Variant& b) {
	return REDUCTION2(reduce_dot, a, b);
}
