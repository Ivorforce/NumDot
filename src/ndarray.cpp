#include "ndarray.h"

#include <godot_cpp/godot.hpp>
#include "xtensor/xtensor.hpp"

#include "xtv.h"
#include "nd.h"
#include "conversion_array.h"

using namespace godot;
using namespace xtv;

void NDArray::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("dtype"), &NDArray::dtype);
	godot::ClassDB::bind_method(D_METHOD("shape"), &NDArray::shape);
	godot::ClassDB::bind_method(D_METHOD("size"), &NDArray::size);
	godot::ClassDB::bind_method(D_METHOD("ndim"), &NDArray::ndim);

	godot::ClassDB::bind_method(D_METHOD("as_type", "type"), &NDArray::as_type);
	godot::ClassDB::bind_method(D_METHOD("to_packed_float32_array"), &NDArray::to_packed_float32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_float64_array"), &NDArray::to_packed_float64_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_byte_array"), &NDArray::to_packed_byte_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int32_array"), &NDArray::to_packed_int32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int64_array"), &NDArray::to_packed_int64_array);
	godot::ClassDB::bind_method(D_METHOD("to_godot_array"), &NDArray::to_godot_array);

	BIND_ENUM_CONSTANT(Float64);
	BIND_ENUM_CONSTANT(Float32);
	BIND_ENUM_CONSTANT(Int8);
	BIND_ENUM_CONSTANT(Int16);
	BIND_ENUM_CONSTANT(Int32);
	BIND_ENUM_CONSTANT(Int64);
	BIND_ENUM_CONSTANT(UInt8);
	BIND_ENUM_CONSTANT(UInt16);
	BIND_ENUM_CONSTANT(UInt32);
	BIND_ENUM_CONSTANT(UInt64);
}

NDArray::NDArray() {
}

NDArray::~NDArray() {
}

String NDArray::_to_string() const {
	return std::visit([](auto& arg){ return xt_to_string(arg); }, *array);
}

NDArray::DType NDArray::dtype() {
	return xtv::dtype(*array);
}

PackedInt64Array NDArray::shape() {
	auto shape = xtv::shape(*array);
	// TODO This seems a bit weird, but it works for now.
	auto packed = PackedInt64Array();
	for (auto d : shape) {
		packed.append(d);
	}
	return packed;
}

uint64_t NDArray::size() {
	return xtv::size(*array);
}

uint64_t NDArray::ndim() {
	return xtv::dimension(*array);
}

Variant NDArray::as_type(DType dtype) {
	return nd::as_type(this, dtype);
}

PackedFloat32Array NDArray::to_packed_float32_array() {
	return xtvariant_to_packed<PackedFloat32Array>(*array);
}

PackedFloat64Array NDArray::to_packed_float64_array() {
	return xtvariant_to_packed<PackedFloat64Array>(*array);
}

PackedByteArray NDArray::to_packed_byte_array() {
	return xtvariant_to_packed<PackedByteArray>(*array);
}

PackedInt32Array NDArray::to_packed_int32_array() {
	return xtvariant_to_packed<PackedInt32Array>(*array);
}

PackedInt64Array NDArray::to_packed_int64_array() {
	return xtvariant_to_packed<PackedInt64Array>(*array);
}

Array NDArray::to_godot_array() {
	return xtvariant_to_godot_array(*array);
}
