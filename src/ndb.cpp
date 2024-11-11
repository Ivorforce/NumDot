#include "ndb.hpp"

#include <vatensor/reduce.hpp>                  // for all, any
#include <stdexcept>                          // for runtime_error
#include <type_traits>                        // for decay_t
#include <utility>                            // for forward
#include <vatensor/comparison.hpp>

#include "gdconvert/conversion_array.hpp"       // for variant_as_array
#include "godot_cpp/core/class_db.hpp"        // for D_METHOD, ClassDB, Meth...
#include "godot_cpp/core/error_macros.hpp"    // for ERR_FAIL_V_MSG
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "vatensor/varray.hpp"                  // for VArray (ptr only), VScalar


using namespace godot;

void ndb::_bind_methods() {
	godot::ClassDB::bind_static_method("ndb", D_METHOD("all", "a"), &ndb::all);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("any", "a"), &ndb::any);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("array_equal", "a", "b"), &ndb::array_equal);
	godot::ClassDB::bind_static_method("ndb", D_METHOD("all_close", "a", "b", "rtol", "atol", "equal_nan"), &ndb::all_close, DEFVAL(1e-05), DEFVAL(1e-08), DEFVAL(false));
}

ndb::ndb() = default;
ndb::~ndb() = default;

template<typename Visitor, typename... Args>
inline bool reduction(Visitor&& visitor, const Args&... args) {
	try {
		const auto result = std::forward<Visitor>(visitor)(*variant_as_array(args)...);

		if constexpr (std::is_same_v<std::decay_t<decltype(result)>, va::VScalar>) {
			return va::static_cast_scalar<bool>(result);
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
	reduction([](const va::VArray& array) { return va::func(array.data); }, (varray1))

#define REDUCTION2(func, varray1, varray2) \
	reduction([](const va::VArray& x1, const va::VArray& x2) { return va::func(x1.data, x2.data); }, (varray1), (varray2))

bool ndb::all(const Variant& a) {
	return REDUCTION1(all, a);
}

bool ndb::any(const Variant& a) {
	return REDUCTION1(any, a);
}

bool ndb::array_equal(const Variant& a, const Variant& b) {
	return REDUCTION2(array_equal, a, b);
}

bool ndb::all_close(const Variant& a, const Variant& b, double_t rtol, double_t atol, bool equal_nan) {
	return reduction([rtol, atol, equal_nan](const va::VArray& a, const va::VArray& b) {
		return va::all_close(a.data, b.data, rtol, atol, equal_nan);
	}, a, b);
}
