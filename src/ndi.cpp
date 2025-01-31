#include "ndi.hpp"

#include <vatensor/linalg.hpp>                  // for reduce_dot
#include <vatensor/reduce.hpp>                  // for max, median, min, norm_l0
#include <cmath>                              // for isinf
#include <stdexcept>                          // for runtime_error
#include <type_traits>                        // for decay_t
#include <utility>                            // for forward
#include <vatensor/xtensor_store.hpp>
#include <vatensor/vfunc/entrypoints.hpp>
#include "gdconvert/conversion_array.hpp"       // for variant_as_array
#include "gdconvert/util.hpp"
#include "godot_cpp/core/class_db.hpp"        // for D_METHOD, ClassDB, Meth...
#include "godot_cpp/core/error_macros.hpp"    // for ERR_FAIL_V_MSG
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "vatensor/varray.hpp"                  // for VArray (ptr only), VScalar


using namespace godot;

void ndi::_bind_methods() {
	godot::ClassDB::bind_static_method("ndi", D_METHOD("sum", "a"), &ndi::sum);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("prod", "a"), &ndi::prod);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("median", "a"), &ndi::median);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("max", "a"), &ndi::max);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("min", "a"), &ndi::min);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("norm", "a", "ord"), &ndi::norm, DEFVAL(2));
	godot::ClassDB::bind_static_method("ndi", D_METHOD("count_nonzero", "a"), &ndi::count_nonzero);
	godot::ClassDB::bind_static_method("ndi", D_METHOD("trace", "v", "offset", "axis1", "axis2"), &ndi::trace, DEFVAL(0), DEFVAL(0), DEFVAL(1));

	godot::ClassDB::bind_static_method("ndi", D_METHOD("reduce_dot", "a", "b"), &ndi::reduce_dot);
}

ndi::ndi() = default;
ndi::~ndi() = default;

#define REDUCTION1(func, varray1) \
	numdot::reduction<int64_t>([](const va::VArray& array) { return va::func(array.data); }, (varray1))

#define REDUCTION1_NEW(func, varray1) \
	numdot::reduction_new<int64_t>([](const va::VArrayTarget& target, const va::VArray& array) { return va::func(va::store::default_allocator, target, array.data, nullptr); }, (varray1))

#define REDUCTION2(func, varray1, varray2) \
	numdot::reduction<int64_t>([](const va::VArray& x1, const va::VArray& x2) { return va::func(x1.data, x2.data); }, (varray1), (varray2))

int64_t ndi::sum(const Variant& a) {
	return REDUCTION1_NEW(sum, a);
}

int64_t ndi::prod(const Variant& a) {
	return REDUCTION1_NEW(prod, a);
}

int64_t ndi::median(const Variant& a) {
	return REDUCTION1(median, a);
}

int64_t ndi::max(const Variant& a) {
	return REDUCTION1_NEW(max, a);
}

int64_t ndi::min(const Variant& a) {
	return REDUCTION1_NEW(min, a);
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

int64_t ndi::count_nonzero(const Variant& a) {
	return numdot::reduction<int64_t>([](const va::VArray& array) {
		return va::count_nonzero(va::store::default_allocator, array.data);
	}, a);
}

int64_t ndi::trace(const Variant& v, int64_t offset, int64_t axis1, int64_t axis2) {
	return numdot::reduction<int64_t>([offset, axis1, axis2](const va::VArray& array) {
		return va::trace_to_scalar(array, offset, axis1, axis2);
	}, v);
}

int64_t ndi::reduce_dot(const Variant& a, const Variant& b) {
	return REDUCTION2(reduce_dot, a, b);
}

#undef REDUCTION1
#undef REDUCTION2
