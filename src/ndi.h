#ifndef NUMDOT_NDI_H
#define NUMDOT_NDI_H

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

class ndi : public Object {
	GDCLASS(ndi, Object)

protected:
	static void _bind_methods();

public:
	ndi();
	~ndi();

	// Reductions.
	static int64_t sum(const Variant& a);
	static int64_t prod(const Variant& a);
	static int64_t median(const Variant& a);
	static int64_t max(const Variant& a);
	static int64_t min(const Variant& a);
	static int64_t norm(const Variant& a, const Variant& ord);

	// Linalg.
	static int64_t reduce_dot(const Variant& a, const Variant& b);
};

#endif
