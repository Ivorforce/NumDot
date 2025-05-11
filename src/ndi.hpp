#ifndef NUMDOT_NDI_H
#define NUMDOT_NDI_H

#include "godot_cpp/classes/object.hpp"   // for Object
#include "godot_cpp/classes/wrapped.hpp"  // for GDCLASS
#include "godot_cpp/variant/variant.hpp"  // for Variant

namespace godot {
	class ClassDB;
}

using namespace godot;

class ndi : public Object {
	GDCLASS(ndi, Object)

protected:
	static void _bind_methods();

public:
	ndi() {
		ERR_FAIL_MSG("This class should not be constructed. It's just a namespace.");
	}

	// Reductions.
	static int64_t sum(const Variant& a);
	static int64_t prod(const Variant& a);
	static int64_t median(const Variant& a);
	static int64_t max(const Variant& a);
	static int64_t min(const Variant& a);
	static int64_t norm(const Variant& a, const Variant& ord);
	static int64_t count_nonzero(const Variant& a);
	static int64_t trace(const Variant& v, int64_t offset, int64_t axis1, int64_t axis2);

	// Linalg.
	static int64_t sum_product(const Variant& a, const Variant& b);
};

#endif
