#include "ndarray.h"

#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "xtv.h"
#include "nd.h"

using namespace godot;
using namespace xtv;

void NDArray::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("dtype"), &NDArray::dtype);
	godot::ClassDB::bind_method(D_METHOD("shape"), &NDArray::shape);
	godot::ClassDB::bind_method(D_METHOD("size"), &NDArray::size);
	godot::ClassDB::bind_method(D_METHOD("ndim"), &NDArray::ndim);

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
	return NDArray::DType((*array).index());
}

PackedInt64Array NDArray::shape() {
	auto shape = xtv::shape(*array);
	// TODO This seems a bit weird, but it works for now.
	auto array = PackedInt64Array();
	for (auto d : shape) {
		array.append(d);
	}
	return array;
}

uint64_t NDArray::size() {
	return xtv::size(*array);
}

uint64_t NDArray::ndim() {
	return xtv::dimension(*array);
}
