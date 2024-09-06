#ifndef NUMDOT_NDARRAY_H
#define NUMDOT_NDARRAY_H

#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"

#ifdef WIN32
#include <windows.h>
#endif

#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/binder_common.hpp>

#include "xtv.h"

using namespace godot;

class NDArray : public RefCounted {
	GDCLASS(NDArray, RefCounted)

private:

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	// We need a shared pointer because things like asarray can return either a view or an array
	std::shared_ptr<xtv::XTVariant> array;

	NDArray();
	NDArray(std::shared_ptr<xtv::XTVariant> array) : array(std::move(array)) {};
	~NDArray();

	xtv::DType dtype();
	PackedInt64Array shape();
	uint64_t size();
	uint64_t ndim();
	
	// Subscript not available, i think. See object's set_bind / get_bind:
	// I think godot assumes that all [] accesses are keypaths.
	// https://github.com/godotengine/godot/blob/514c564a8c855d798ec6b5a52860e5bca8d57bc9/core/object/object.h#L643
	void set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	Variant get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	double_t get_float(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	int64_t get_int(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);

	Variant as_type(xtv::DType dtype);

	double_t to_float();
	int64_t to_int();

	PackedFloat32Array to_packed_float32_array();
	PackedFloat64Array to_packed_float64_array();
	PackedByteArray to_packed_byte_array();
	PackedInt32Array to_packed_int32_array();
	PackedInt64Array to_packed_int64_array();

	Array to_godot_array();
};

#endif
