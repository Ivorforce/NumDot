#ifndef NUMDOT_NDARRAY_H
#define NUMDOT_NDARRAY_H

#ifdef WIN32
#include <windows.h>
#endif

#include <math.h>                                      // for double_t
#include <stdint.h>                                    // for uint64_t, int64_t
#include <godot_cpp/classes/ref_counted.hpp>           // for RefCounted
#include <godot_cpp/variant/variant.hpp>               // for Variant
#include <utility>                                     // for move
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
#include "vatensor/varray.h"                                    // for DType, VArray
namespace godot { class ClassDB; }

using namespace godot;

class NDArray : public RefCounted {
	GDCLASS(NDArray, RefCounted)

private:

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	// We need a shared pointer because things like asarray can return either a view or an array
	va::VArray array;

	NDArray();
	explicit NDArray(va::VArray array) : array(std::move(array)) {};
	~NDArray() override;

	[[nodiscard]] va::DType dtype() const;
	[[nodiscard]] PackedInt64Array shape() const;
	[[nodiscard]] uint64_t size() const;
	[[nodiscard]] uint64_t array_size_in_bytes() const;
	[[nodiscard]] uint64_t ndim() const;
	
	// Subscript not available, i think. See object's set_bind / get_bind:
	// I think godot assumes that all [] accesses are keypaths.
	// https://github.com/godotengine/godot/blob/514c564a8c855d798ec6b5a52860e5bca8d57bc9/core/object/object.h#L643
	void set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	Ref<NDArray> get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	double_t get_float(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	int64_t get_int(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);

	[[nodiscard]] Variant as_type(va::DType dtype) const;

	[[nodiscard]] double_t to_float() const;
	[[nodiscard]] int64_t to_int() const;

	[[nodiscard]] PackedFloat32Array to_packed_float32_array() const;
	[[nodiscard]] PackedFloat64Array to_packed_float64_array() const;
	[[nodiscard]] PackedByteArray to_packed_byte_array() const;
	[[nodiscard]] PackedInt32Array to_packed_int32_array() const;
	[[nodiscard]] PackedInt64Array to_packed_int64_array() const;

	[[nodiscard]] Array to_godot_array() const;
};

#endif
