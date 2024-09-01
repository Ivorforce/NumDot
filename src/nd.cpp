#include "nd.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <iostream>
#include "xtensor/xadapt.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xlayout.hpp"

using namespace godot;

void ND::_bind_methods() {
	godot::ClassDB::bind_static_method("ND", D_METHOD("asarray", "array"), &ND::asarray);
	godot::ClassDB::bind_static_method("ND", D_METHOD("zeros", "shape", "dtype"), &ND::zeros, DEFVAL(nullptr), DEFVAL(NDArray::DType::Double));
	godot::ClassDB::bind_static_method("ND", D_METHOD("ones", "shape", "dtype"), &ND::ones, DEFVAL(nullptr), DEFVAL(NDArray::DType::Double));

	godot::ClassDB::bind_static_method("ND", D_METHOD("add", "a", "b"), &ND::add);
	godot::ClassDB::bind_static_method("ND", D_METHOD("subtract", "a", "b"), &ND::subtract);
	godot::ClassDB::bind_static_method("ND", D_METHOD("multiply", "a", "b"), &ND::multiply);
	godot::ClassDB::bind_static_method("ND", D_METHOD("divide", "a", "b"), &ND::divide);
}

ND::ND() {
}

ND::~ND() {
	// Add your cleanup here.
}

template <typename T>
bool _asshape(Variant shape, T &target) {
	auto type = shape.get_type();

	if (Variant::can_convert(type, Variant::Type::INT)) {
		target = { uint64_t(shape) };
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_INT32_ARRAY)) {
		auto shape_array = PackedInt32Array(shape);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		target = xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape);
		return true;
	}

	ERR_FAIL_V_MSG(false, "Variant cannot be converted to a shape.");
}

bool _asarray(Variant array, std::shared_ptr<NDArrayVariant> &target) {
	auto type = array.get_type();

	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			target = ndarray->array;
			return true;
		}
	}

	if (Variant::can_convert(type, Variant::Type::INT)) {
		// TODO Int array
		target = std::make_shared<NDArrayVariant>(xt::xarray<double>());
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::FLOAT)) {
		target = std::make_shared<NDArrayVariant>(xt::xarray<double>(float_t(array)));
		return true;
	}
	if (Variant::can_convert(type, Variant::Type::PACKED_FLOAT32_ARRAY)) {
		auto shape_array = PackedInt32Array(array);
		uint64_t size = shape_array.size();

		xt::static_shape<std::size_t, 1> shape_of_shape = { size };

		target = std::make_shared<NDArrayVariant>(
			xt::xarray<double>(xt::adapt(shape_array.ptrw(), size, xt::no_ownership(), shape_of_shape))
		);
		return true;
	}

	ERR_FAIL_V_MSG(false, "Variant cannot be converted to an array.");
}

Variant ND::asarray(Variant array) {
	auto type = array.get_type();

	if (type == Variant::OBJECT) {
		if (auto ndarray = dynamic_cast<NDArray*>((Object*)(array))) {
			return array;
		}
	}

	NDArray *result = memnew(NDArray());
	if (!_asarray(array, result->array)) {
		return nullptr;
	}

	return Variant(result);
}

// TODO This should use templates, but i couldn't get it to work.
#define DTypeSwitch(code) switch (dtype) {\
	case NDArray::DType::Double:\
		(*result).emplace<xt::xarray<double_t>>(code<double_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::Float:\
		(*result).emplace<xt::xarray<float_t>>(code<float_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::Int8:\
		(*result).emplace<xt::xarray<int8_t>>(code<int8_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::Int16:\
		(*result).emplace<xt::xarray<int16_t>>(code<int16_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::Int32:\
		(*result).emplace<xt::xarray<int32_t>>(code<int32_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::Int64:\
		(*result).emplace<xt::xarray<int64_t>>(code<int64_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::UInt8:\
		(*result).emplace<xt::xarray<uint8_t>>(code<uint8_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::UInt16:\
		(*result).emplace<xt::xarray<uint16_t>>(code<uint16_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::UInt32:\
		(*result).emplace<xt::xarray<uint32_t>>(code<uint32_t>(std::move(shape_array)));\
		break;\
	case NDArray::DType::UInt64:\
		(*result).emplace<xt::xarray<uint64_t>>(code<uint64_t>(std::move(shape_array)));\
		break;\
}

Variant ND::zeros(Variant shape, NDArray::DType dtype) {
	xt::xarray<size_t> shape_array;
	if (!_asshape(shape, shape_array)) {
		return nullptr;
	}

	// General note: By creating the object first, and assigning later,
	//  we avoid creating the result on the stack first and copying to the heap later.
	// This means this kind of ugly contraption is quite a lot fasterhat than the alternative.
	auto result = std::make_shared<NDArrayVariant>();

	DTypeSwitch(xt::zeros);

	return Variant(memnew(NDArray(result)));
}

Variant ND::ones(Variant shape, NDArray::DType dtype) {
	xt::xarray<size_t> shape_array;
	if (!_asshape(shape, shape_array)) {
		return nullptr;
	}

	auto result = std::make_shared<NDArrayVariant>();

	DTypeSwitch(xt::ones);

	return Variant(memnew(NDArray(result)));
}

template <typename operation>
struct BinOperation {
	template<typename A, typename B>
	NDArray *operator()(xt::xarray<A>& a, xt::xarray<B>& b) const {
		// ResultType = what results from the usual C++ common promotion of a + b.
		using ResultType = typename std::common_type<A, B>::type;

		// General note: By creating the object first, and assigning later,
		//  we avoid creating the result on the stack first and copying to the heap later.
		// This means this kind of ugly contraption is quite a lot faster than the alternative.
		auto result = std::make_shared<NDArrayVariant>(xt::xarray<ResultType>());
		
		// Run the operation itself.
		std::get<xt::xarray<ResultType>>(*result) = operation()(a, b);

		// Assign to the result array.
		return memnew(NDArray(result));
	}
};

template <typename operation>
inline Variant bin_op(Variant a, Variant b) {
	std::shared_ptr<NDArrayVariant> a_;
	if (!_asarray(a, a_)) {
		return nullptr;
	}
	std::shared_ptr<NDArrayVariant> b_;
	if (!_asarray(b, b_)) {
		return nullptr;
	}

	return Variant(std::visit(BinOperation<operation>{}, *a_, *b_));
}

Variant ND::add(Variant a, Variant b) {
	// godot::UtilityFunctions::print(xt::has_simd_interface<xt::xarray<int64_t>>::value);
	// godot::UtilityFunctions::print(xt::has_simd_type<xt::xarray<int64_t>>::value);
	return bin_op<std::plus<xt::xarray<double>>>(a, b);
}

Variant ND::subtract(Variant a, Variant b) {
	return bin_op<std::minus<xt::xarray<double>>>(a, b);
}

Variant ND::multiply(Variant a, Variant b) {
	return bin_op<std::multiplies<xt::xarray<double>>>(a, b);
}

Variant ND::divide(Variant a, Variant b) {
	return bin_op<std::divides<xt::xarray<double>>>(a, b);
}
