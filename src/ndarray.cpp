#include "ndarray.h"

#include <godot_cpp/godot.hpp>
#include "xtensor/xtensor.hpp"

#include "xtv.h"
#include "nd.h"
#include "conversion_array.h"
#include "conversion_slice.h"

using namespace godot;

void NDArray::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("dtype"), &NDArray::dtype);
	godot::ClassDB::bind_method(D_METHOD("shape"), &NDArray::shape);
	godot::ClassDB::bind_method(D_METHOD("size"), &NDArray::size);
	godot::ClassDB::bind_method(D_METHOD("ndim"), &NDArray::ndim);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get", &NDArray::get);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "set", &NDArray::set);

	godot::ClassDB::bind_method(D_METHOD("as_type", "type"), &NDArray::as_type);
	godot::ClassDB::bind_method(D_METHOD("to_packed_float32_array"), &NDArray::to_packed_float32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_float64_array"), &NDArray::to_packed_float64_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_byte_array"), &NDArray::to_packed_byte_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int32_array"), &NDArray::to_packed_int32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int64_array"), &NDArray::to_packed_int64_array);
	godot::ClassDB::bind_method(D_METHOD("to_godot_array"), &NDArray::to_godot_array);
}

NDArray::NDArray() {
}

NDArray::~NDArray() {
}

String NDArray::_to_string() const {
	return std::visit([](auto& arg){ return xt_to_string(arg); }, *array);
}

nd::DType NDArray::dtype() {
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

Variant NDArray::as_type(nd::DType dtype) {
	return nd::as_type(this, dtype);
}

void NDArray::set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	if (arg_count < 1) {
		ERR_FAIL_MSG("First argument (value) must be set. Ignoring assignment.");
	}

	try {
		const Variant &value = *args[0];

		xt::xstrided_slice_vector sv(arg_count - 1);
		for (int i = 1; i < arg_count; i++) {
			sv[i - 1] = variant_as_slice_part(*args[i]);
		}

		switch (value.get_type()) {
			case Variant::INT:
				xtv::set_slice_value(*array, sv, int64_t(value));
				return;
			case Variant::FLOAT:
				xtv::set_slice_value(*array, sv, double_t(value));
				return;
			// TODO We could optimize more assignments of literals.
			//  Just need to figure out how, ideally without duplicating code - as_array already does much type checking work.
			default:
				std::shared_ptr<xtv::XTVariant> a_;
				if (!variant_as_array(value, a_)) {
					return;
				}
				xtv::set_slice(*array, sv, *a_);
				return;
		}
	}
	catch (std::runtime_error error) {
		ERR_FAIL_MSG(error.what());
	}
}

Variant NDArray::get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		xt::xstrided_slice_vector sv(arg_count);
		for (int i = 0; i < arg_count; i++) {
			sv[i] = variant_as_slice_part(*args[i]);
		}

		auto result = xtv::get_slice(*array, sv);
		return Variant(memnew(NDArray(result)));
	}
	catch (std::runtime_error error) {
		ERR_FAIL_V_MSG(nullptr, error.what());
	}
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
