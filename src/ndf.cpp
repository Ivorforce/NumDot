#include "ndf.h"

#include <vatensor/linalg.h>                  // for reduce_dot
#include <vatensor/reduce.h>                  // for max, mean, median, min
#include <cmath>                              // for double_t, isinf
#include <cstdint>                            // for int64_t
#include <stdexcept>                          // for runtime_error
#include <type_traits>                        // for decay_t
#include <utility>                            // for forward
#include "gdconvert/conversion_array.h"       // for variant_as_array
#include "godot_cpp/core/class_db.hpp"        // for D_METHOD, ClassDB, Meth...
#include "godot_cpp/core/error_macros.hpp"    // for ERR_FAIL_V_MSG
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "vatensor/varray.h"                  // for VArray (ptr only), VScalar

using namespace godot;

void ndf::_bind_methods() {
	godot::ClassDB::bind_static_method("ndf", D_METHOD("sum", "a"), &ndf::sum);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("prod", "a"), &ndf::prod);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("mean", "a"), &ndf::mean);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("median", "a"), &ndf::median);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("var", "a"), &ndf::var);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("std", "a"), &ndf::std);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("max", "a"), &ndf::max);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("min", "a"), &ndf::min);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("norm", "a", "ord"), &ndf::norm, DEFVAL(nullptr), DEFVAL(2));

	godot::ClassDB::bind_static_method("ndf", D_METHOD("reduce_dot", "a", "b"), &ndf::reduce_dot);
}

ndf::ndf() = default;
ndf::~ndf() = default;

template <typename Visitor, typename... Args>
inline double_t reduction(Visitor&& visitor, const Args&... args) {
	try {
		const auto result = std::forward<Visitor>(visitor)(variant_as_array(args)...);

		if constexpr (std::is_same_v<std::decay_t<decltype(result)>, va::VScalar>) {
			return va::scalar_to_type<double_t>(result);
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

double_t ndf::sum(const Variant& a) {
	return REDUCTION1(sum, a);
}

double_t ndf::prod(const Variant& a) {
	return REDUCTION1(prod, a);
}

double_t ndf::mean(const Variant& a) {
	return REDUCTION1(mean, a);
}

double_t ndf::median(const Variant &a) {
	return REDUCTION1(median, a);
}

double_t ndf::var(const Variant& a) {
	return REDUCTION1(var, a);
}

double_t ndf::std(const Variant& a) {
	return REDUCTION1(std, a);
}

double_t ndf::max(const Variant& a) {
	return REDUCTION1(max, a);
}

double_t ndf::min(const Variant& a) {
	return REDUCTION1(min, a);
}

double_t ndf::norm(const Variant& a, const Variant& ord) {
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
			if (std::isinf(static_cast<double_t>(ord))) {
				return REDUCTION1(norm_linf, a);
			}
		default:
			break;
	}

	ERR_FAIL_V_MSG({}, "This norm is currently not supported");
}

//double_t ndf::all(const Variant& a) {
//    return REDUCTION1(all, a);
//}
//
//double_t ndf::any(const Variant& a) {
//    return REDUCTION1(any, a);
//}

double_t ndf::reduce_dot(const Variant& a, const Variant& b) {
	return REDUCTION2(reduce_dot, a, b);
}
