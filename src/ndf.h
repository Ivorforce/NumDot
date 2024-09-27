#ifndef NUMDOT_NDF_H
#define NUMDOT_NDF_H

#include "vatensor/auto_defines.h"
#include <cstdint>                            // for int64_t, uint64_t
#include <godot_cpp/classes/ref.hpp>          // for Ref
#include <godot_cpp/core/binder_common.hpp>   // for VARIANT_ENUM_CAST
#include "godot_cpp/classes/object.hpp"       // for Object
#include "godot_cpp/classes/wrapped.hpp"      // for GDCLASS
#include "godot_cpp/core/class_db.hpp"        // for ClassDB (ptr only), DEFVAL
#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "godot_cpp/variant/variant.hpp"      // for Variant
#include "ndarray.h"                          // for NDArray
#include "vatensor/varray.h"                           // for DType


using namespace godot;

class ndf : public Object {
	GDCLASS(ndf, Object)

protected:
	static void _bind_methods();

public:
	ndf();
	~ndf();

	// Reductions.
	static double_t sum(const Variant& a);
	static double_t prod(const Variant& a);
	static double_t mean(const Variant& a);
	static double_t median(const Variant& a);
	static double_t var(const Variant& a);
	static double_t std(const Variant& a);
	static double_t max(const Variant& a);
	static double_t min(const Variant& a);
	static double_t norm(const Variant& a, const Variant& ord);

//	// Logical.
//	static bool all(const Variant& a);
//	static bool any(const Variant& a);

	// Linalg.
	static double_t reduce_dot(const Variant& a, const Variant& b);
};

#endif
