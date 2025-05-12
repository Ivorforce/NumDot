#include "ndf.hpp"

#include <vatensor/linalg.hpp>                  // for sum_product
#include <cmath>                              // for double_t, isinf
#include <cstdint>                            // for int64_t
#include <stdexcept>                          // for runtime_error
#include <type_traits>                        // for decay_t
#include <utility>                            // for forward
#include <vatensor/xtensor_store.hpp>
#include "gdconvert/conversion_array.hpp"       // for variant_as_array
#include "gdconvert/util.hpp"
#include "godot_cpp/core/class_db.hpp"        // for D_METHOD, ClassDB, Meth...
#include "godot_cpp/core/error_macros.hpp"    // for ERR_FAIL_V_MSG
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "vatensor/varray.hpp"                  // for VArray (ptr only), VScalar
#include "vatensor/vfunc/entrypoints.hpp"                  // for VArray (ptr only), VScalar

using namespace godot;

void ndf::_bind_methods() {
	godot::ClassDB::bind_static_method("ndf", D_METHOD("sum", "a"), &ndf::sum);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("prod", "a"), &ndf::prod);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("mean", "a"), &ndf::mean);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("median", "a"), &ndf::median);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("var", "a"), &ndf::variance);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("std", "a"), &ndf::standard_deviation);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("max", "a"), &ndf::max);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("min", "a"), &ndf::min);
	godot::ClassDB::bind_static_method("ndf", D_METHOD("norm", "a", "ord"), &ndf::norm, DEFVAL(2));
	godot::ClassDB::bind_static_method("ndf", D_METHOD("trace", "v", "offset", "axis1", "axis2"), &ndf::trace, DEFVAL(0), DEFVAL(0), DEFVAL(1));

	godot::ClassDB::bind_static_method("ndf", D_METHOD("sum_product", "a", "b"), &ndf::sum_product);
}

#define REDUCTION1(func, varray1) \
numdot::reduction_new<double_t>([](const va::VArrayTarget& target, const va::VArray& array) { va::func(va::store::default_allocator, target, array.data, nullptr); }, (varray1))

#define REDUCTION2(func, varray1, varray2) \
numdot::reduction_new<double_t>([](const va::VArrayTarget& target, const va::VArray& x1, const va::VArray& x2) { va::func(va::store::default_allocator, target, x1.data, x2.data, nullptr); }, (varray1), (varray2))

double_t ndf::sum(const Variant& a) {
	return REDUCTION1(sum, a);
}

double_t ndf::prod(const Variant& a) {
	return REDUCTION1(prod, a);
}

double_t ndf::mean(const Variant& a) {
	return REDUCTION1(mean, a);
}

double_t ndf::median(const Variant& a) {
	return REDUCTION1(median, a);
}

double_t ndf::variance(const Variant& a) {
	return REDUCTION1(variance, a);
}

double_t ndf::standard_deviation(const Variant& a) {
	return REDUCTION1(standard_deviation, a);
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

double_t ndf::trace(const Variant& v, int64_t offset, int64_t axis1, int64_t axis2) {
	return numdot::reduction_new<double_t>([offset, axis1, axis2](const va::VArrayTarget& target, const va::VArray& array) {
		return va::trace(va::store::default_allocator, target, array, offset, axis1, axis2);
	}, v);
}

double_t ndf::sum_product(const Variant& a, const Variant& b) {
	return REDUCTION2(sum_product, a, b);
}

#undef REDUCTION1
#undef REDUCTION2
