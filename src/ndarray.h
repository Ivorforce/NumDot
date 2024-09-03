#ifndef NUMDOT_NDARRAY_H
#define NUMDOT_NDARRAY_H

#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"

#ifdef WIN32
#include <windows.h>
#endif

#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/binder_common.hpp>

#include "xtv.h"

using namespace godot;

class NDArray : public Object {
	GDCLASS(NDArray, Object)

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
	
	Variant as_type(xtv::DType dtype);
	PackedFloat32Array to_packed_float32_array();
	PackedFloat64Array to_packed_float64_array();
	PackedByteArray to_packed_byte_array();
	PackedInt32Array to_packed_int32_array();
	PackedInt64Array to_packed_int64_array();
	Array to_godot_array();
};

#endif
