#ifndef NUMDOT_NDRANDOMGENERATOR_H
#define NUMDOT_NDRANDOMGENERATOR_H

#ifdef WIN32
#include <windows.h>
#endif

#include "vatensor/auto_defines.hpp"

#include <cmath>                                       // for Layout
#include <cstdint>                                     // for uint64_t, int64_t
#include <godot_cpp/classes/ref_counted.hpp>           // for RefCounted
#include <godot_cpp/variant/variant.hpp>               // for Variant
#include <utility>                                     // for move
#include <memory>                                      // for make_shared, shared_ptr
#include "gdextension_interface.h"                     // for GDExtensionCal...
#include "godot_cpp/classes/ref.hpp"                   // for Ref
#include "godot_cpp/classes/wrapped.hpp"               // for GDCLASS
#include "godot_cpp/variant/array.hpp"                 // for Array
#include "godot_cpp/variant/packed_byte_array.hpp"     // for PackedByteArray
#include "godot_cpp/variant/packed_float32_array.hpp"  // for PackedFloat32A...
#include "godot_cpp/variant/packed_float64_array.hpp"  // for PackedFloat64A...
#include "godot_cpp/variant/packed_int32_array.hpp"    // for PackedInt32Array
#include "godot_cpp/variant/packed_int64_array.hpp"    // for PackedInt64Array
#include "godot_cpp/variant/string.hpp"                // for String
#include "godot_cpp/variant/color.hpp"                 // for Color
#include "godot_cpp/variant/packed_color_array.hpp"    // for PackedColorArray
#include "godot_cpp/variant/packed_vector2_array.hpp"  // for PackedVector2A...
#include "godot_cpp/variant/packed_vector3_array.hpp"  // for PackedVector3A...
#include "godot_cpp/variant/packed_vector4_array.hpp"  // for PackedVector4A...
#include "godot_cpp/variant/typed_array.hpp"           // for TypedArray
#include "godot_cpp/variant/vector2.hpp"               // for Vector2
#include "godot_cpp/variant/vector2i.hpp"              // for Vector2i
#include "godot_cpp/variant/vector3.hpp"               // for Vector3
#include "godot_cpp/variant/vector3i.hpp"              // for Vector3i
#include "godot_cpp/variant/vector4.hpp"               // for Vector4
#include "godot_cpp/variant/vector4i.hpp"              // for Vector4i
#include "vatensor/varray.hpp"                                    // for DType, VArray
#include "vatensor/vrandom.hpp"                                    // for DType, VArray
#include "ndarray.hpp"

namespace godot {
	class ClassDB;
}

using namespace godot;

class NDRandomGenerator : public RefCounted {
	GDCLASS(NDRandomGenerator, RefCounted)

private:

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	// We need a shared pointer because things like asarray can return either a view or an array
	va::random::VRandomEngine engine;

	NDRandomGenerator();
	explicit NDRandomGenerator(va::random::VRandomEngine&& engine) : engine(std::forward<va::random::VRandomEngine>(engine)) {};
	~NDRandomGenerator() override;

	TypedArray<NDRandomGenerator> spawn(int64_t n);

	Ref<NDArray> random(const Variant& shape = PackedByteArray(), va::DType dtype = va::Float64);
	Ref<NDArray> integers(int64_t low_or_high, const Variant& high, const Variant& shape = PackedByteArray(), va::DType dtype = va::Float64, bool endpoint = false);
	Ref<NDArray> randn(const Variant& shape = PackedByteArray(), va::DType dtype = va::Float64);
};

#endif
