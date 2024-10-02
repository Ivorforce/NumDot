#ifndef NUMDOT_NDB_H
#define NUMDOT_NDB_H

#include "godot_cpp/classes/object.hpp"   // for Object
#include "godot_cpp/classes/wrapped.hpp"  // for GDCLASS
#include "godot_cpp/variant/variant.hpp"  // for Variant


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
