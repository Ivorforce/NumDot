#ifndef NUMDOT_NDF_H
#define NUMDOT_NDF_H

#include <cmath>                         // for double_t
#include "godot_cpp/classes/object.hpp"   // for Object
#include "godot_cpp/classes/wrapped.hpp"  // for GDCLASS
#include "godot_cpp/variant/variant.hpp"  // for Variant

namespace godot {
	class ClassDB;
}

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
	static double_t trace(const Variant& v, int64_t offset, int64_t axis1, int64_t axis2);

	//	// Logical.
	//	static bool all(const Variant& a);
	//	static bool any(const Variant& a);

	// Linalg.
	static double_t reduce_dot(const Variant& a, const Variant& b);
};

#endif
