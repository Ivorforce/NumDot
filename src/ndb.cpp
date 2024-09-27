#include "ndb.h"

#include <vatensor/reduce.h>                // for max, mean, min, prod, std
#include <cmath>                            // for double_t
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

void ndb::_bind_methods() {
	godot::ClassDB::bind_static_method("ndb", D_METHOD("all", "a"), &ndb::all);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("any", "a"), &ndb::any);
}

ndb::ndb() = default;
ndb::~ndb() = default;

template <typename Visitor, typename... Args>
inline bool reduction(Visitor&& visitor, const Args&... args) {
	try {
		const auto result = std::forward<Visitor>(visitor)(variant_as_array(args)...);

		if constexpr (std::is_same_v<std::decay_t<decltype(result)>, va::VScalar>) {
			return va::scalar_to_type<bool>(result);
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

bool ndb::all(const Variant& a) {
    return REDUCTION1(all, a);
}

bool ndb::any(const Variant& a) {
    return REDUCTION1(any, a);
}
