#include "ndb.hpp"

#include <vatensor/reduce.hpp>                  // for all, any
#include <stdexcept>                          // for runtime_error
#include <type_traits>                        // for decay_t
#include <utility>                            // for forward
#include <vatensor/comparison.hpp>
#include <vatensor/xtensor_store.hpp>
#include <vatensor/vfunc/entrypoints.hpp>
#include "gdconvert/conversion_array.hpp"       // for variant_as_array
#include "gdconvert/util.hpp"
#include "godot_cpp/core/class_db.hpp"        // for D_METHOD, ClassDB, Meth...
#include "godot_cpp/core/error_macros.hpp"    // for ERR_FAIL_V_MSG
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "vatensor/varray.hpp"                  // for VArray (ptr only), VScalar


using namespace godot;

void ndb::_bind_methods() {
	godot::ClassDB::bind_static_method("ndb", D_METHOD("all", "a"), &ndb::all);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("any", "a"), &ndb::any);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("array_equiv", "a", "b"), &ndb::array_equiv);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("array_equal", "a", "b"), &ndb::array_equal);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("all_close", "a", "b", "rtol", "atol", "equal_nan"), &ndb::all_close, DEFVAL(1e-05), DEFVAL(1e-08), DEFVAL(false));
}

ndb::ndb() = default;
ndb::~ndb() = default;

#define REDUCTION1(func, varray1) \
numdot::reduction<bool>([](const va::VArray& array) { return va::func(array.data); }, (varray1))

#define REDUCTION1_NEW(func, varray1) \
numdot::reduction_new<bool>([](const va::VArrayTarget& target, const va::VArray& array) { return va::func(va::store::default_allocator, target, array.data, nullptr); }, (varray1))

#define REDUCTION2(func, varray1, varray2) \
numdot::reduction<bool>([](const va::VArray& x1, const va::VArray& x2) { return va::func(x1.data, x2.data); }, (varray1), (varray2))

bool ndb::all(const Variant& a) {
	return REDUCTION1_NEW(all, a);
}

bool ndb::any(const Variant& a) {
	return REDUCTION1_NEW(any, a);
}

bool ndb::array_equiv(const Variant& a, const Variant& b) {
	return REDUCTION2(array_equiv, a, b);
}

bool ndb::array_equal(const Variant& a, const Variant& b) {
	return REDUCTION2(array_equal, a, b);
}

bool ndb::all_close(const Variant& a, const Variant& b, double_t rtol, double_t atol, bool equal_nan) {
	return numdot::reduction<bool>([rtol, atol, equal_nan](const va::VArray& a, const va::VArray& b) {
		return va::all_close(a.data, b.data, rtol, atol, equal_nan);
	}, a, b);
}

#undef REDUCTION1
#undef REDUCTION2
