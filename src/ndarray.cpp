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
	godot::ClassDB::bind_method(D_METHOD("array_size_in_bytes"), &NDArray::array_size_in_bytes);
	godot::ClassDB::bind_method(D_METHOD("ndim"), &NDArray::ndim);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "set", &NDArray::set);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get", &NDArray::get);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get_float", &NDArray::get_float);
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "get_int", &NDArray::get_int);

	godot::ClassDB::bind_method(D_METHOD("as_type", "type"), &NDArray::as_type);

	godot::ClassDB::bind_method(D_METHOD("to_float"), &NDArray::to_float);
	godot::ClassDB::bind_method(D_METHOD("to_int"), &NDArray::to_int);

	godot::ClassDB::bind_method(D_METHOD("to_packed_float32_array"), &NDArray::to_packed_float32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_float64_array"), &NDArray::to_packed_float64_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_byte_array"), &NDArray::to_packed_byte_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int32_array"), &NDArray::to_packed_int32_array);
	godot::ClassDB::bind_method(D_METHOD("to_packed_int64_array"), &NDArray::to_packed_int64_array);
	godot::ClassDB::bind_method(D_METHOD("to_godot_array"), &NDArray::to_godot_array);
}

NDArray::NDArray() = default;

NDArray::~NDArray() = default;

String NDArray::_to_string() const {
	return std::visit([](auto& arg){ return xt_to_string(arg); }, *array);
}

nd::DType NDArray::dtype() const {
	return xtv::dtype(*array);
}

PackedInt64Array NDArray::shape() const {
	auto shape = xtv::shape(*array);
	// TODO This seems a bit weird, but it works for now.
	auto packed = PackedInt64Array();
	for (auto d : shape) {
		packed.append(d);
	}
	return packed;
}

uint64_t NDArray::size() const {
	return xtv::size(*array);
}

uint64_t NDArray::array_size_in_bytes() const {
	return xtv::size_of_array_in_bytes(*array);
}

uint64_t NDArray::ndim() const {
	return xtv::dimension(*array);
}

Variant NDArray::as_type(nd::DType dtype) const {
	return nd::as_type(this, dtype);
}

void NDArray::set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	if (arg_count < 1) {
		ERR_FAIL_MSG("First argument (value) must be set. Ignoring assignment.");
	}

	try {
		const Variant &value = *args[0];
		xt::xstrided_slice_vector sv = variants_as_slice_vector(args + 1, arg_count - 1, error);

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

Ref<NDArray> NDArray::get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		xt::xstrided_slice_vector sv = variants_as_slice_vector(args, arg_count, error);

		auto result = xtv::get_slice(*array, sv);
		return Ref<NDArray>(memnew(NDArray(result)));
	}
	catch (std::runtime_error error) {
		ERR_FAIL_V_MSG(Ref<NDArray>(), error.what());
	}
}

double_t NDArray::get_float(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		xt::xstrided_slice_vector sv = variants_as_slice_vector(args, arg_count, error);

		// TODO This can probably be faster with a specific implementation that doesn't allocate an xarray.
		auto result = xtv::get_slice(*array, sv);
		return xtv::to_single_value<double_t>(*result);
	}
	catch (std::runtime_error error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

int64_t NDArray::get_int(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error) {
	try {
		xt::xstrided_slice_vector sv = variants_as_slice_vector(args, arg_count, error);

		// TODO This can probably be faster with a specific implementation that doesn't allocate an xarray.
		auto result = xtv::get_slice(*array, sv);
		return xtv::to_single_value<int64_t>(*result);
	}
	catch (std::runtime_error error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

double_t NDArray::to_float() const {
	try {
		return xtv::to_single_value<double_t>(*array);
	}
	catch (std::runtime_error error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

int64_t NDArray::to_int() const {
	try {
		return xtv::to_single_value<int64_t>(*array);
	}
	catch (std::runtime_error error) {
		ERR_FAIL_V_MSG(0, error.what());
	}
}

PackedFloat32Array NDArray::to_packed_float32_array() const {
	return xtvariant_to_packed<PackedFloat32Array>(*array);
}

PackedFloat64Array NDArray::to_packed_float64_array() const {
	return xtvariant_to_packed<PackedFloat64Array>(*array);
}

PackedByteArray NDArray::to_packed_byte_array() const {
	return xtvariant_to_packed<PackedByteArray>(*array);
}

PackedInt32Array NDArray::to_packed_int32_array() const {
	return xtvariant_to_packed<PackedInt32Array>(*array);
}

PackedInt64Array NDArray::to_packed_int64_array() const {
	return xtvariant_to_packed<PackedInt64Array>(*array);
}

Array NDArray::to_godot_array() const {
	return xtvariant_to_godot_array(*array);
}
