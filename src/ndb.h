#ifndef NUMDOT_NDB_H
#define NUMDOT_NDB_H

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

class ndb : public Object {
	GDCLASS(ndb, Object)

protected:
	static void _bind_methods();

public:
	ndb();
	~ndb();

	// Logical.
	static bool all(const Variant& a);
	static bool any(const Variant& a);
};

#endif
