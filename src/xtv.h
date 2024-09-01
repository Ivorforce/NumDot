#ifndef NUMDOT_XTV_H
#define NUMDOT_XTV_H

#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"

namespace xtv {

using Variant = std::variant<
	xt::xarray<double_t>,
 	xt::xarray<float_t>,
 	xt::xarray<int8_t>,
 	xt::xarray<int16_t>,
 	xt::xarray<int32_t>,
 	xt::xarray<int64_t>,
 	xt::xarray<uint8_t>,
 	xt::xarray<uint16_t>,
 	xt::xarray<uint32_t>,
 	xt::xarray<uint64_t>
>;

using VariantContainedTypes = std::tuple<
	double_t,
 	float_t,
 	int8_t,
 	int16_t,
 	int32_t,
 	int64_t,
 	uint8_t,
 	uint16_t,
 	uint32_t,
 	uint64_t
>;

enum DType {
    Double,
    Float,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    DTypeMax
};


// TODO This should use templates, but i couldn't get it to work.
#define DTypeSwitch(dtype, code, args) switch (dtype) {\
	case xtv::DType::Double:\
		(*result).emplace<xt::xarray<double_t>>(code<double_t>(args));\
		break;\
	case xtv::DType::Float:\
		(*result).emplace<xt::xarray<float_t>>(code<float_t>(args));\
		break;\
	case xtv::DType::Int8:\
		(*result).emplace<xt::xarray<int8_t>>(code<int8_t>(args));\
		break;\
	case xtv::DType::Int16:\
		(*result).emplace<xt::xarray<int16_t>>(code<int16_t>(args));\
		break;\
	case xtv::DType::Int32:\
		(*result).emplace<xt::xarray<int32_t>>(code<int32_t>(args));\
		break;\
	case xtv::DType::Int64:\
		(*result).emplace<xt::xarray<int64_t>>(code<int64_t>(args));\
		break;\
	case xtv::DType::UInt8:\
		(*result).emplace<xt::xarray<uint8_t>>(code<uint8_t>(args));\
		break;\
	case xtv::DType::UInt16:\
		(*result).emplace<xt::xarray<uint16_t>>(code<uint16_t>(args));\
		break;\
	case xtv::DType::UInt32:\
		(*result).emplace<xt::xarray<uint32_t>>(code<uint32_t>(args));\
		break;\
	case xtv::DType::UInt64:\
		(*result).emplace<xt::xarray<uint64_t>>(code<uint64_t>(args));\
		break;\
	case xtv::DType::DTypeMax:\
		return nullptr;\
}

static std::shared_ptr<Variant> array(Variant &existing_array, DType dtype) {
    auto result = std::make_shared<Variant>();

	// TODO Using the switch here is kinda dumb, but for now it's the easiest way of making it work, making use of std::visit later.
	DTypeSwitch(dtype, xt::xarray, );

	std::visit([](auto& a, auto& b){
		a = b;
	}, *result, existing_array);

    return result;
}

static std::shared_ptr<Variant> zeros(xt::xarray<size_t>& shape_array, DType dtype) {
	// General note: By creating the object first, and assigning later,
	//  we avoid creating the result on the stack first and copying to the heap later.
	// This means this kind of ugly contraption is quite a lot fasterhat than the alternative.
	auto result = std::make_shared<xtv::Variant>();

	DTypeSwitch(dtype, xt::zeros, shape_array);

	return result;
}

static std::shared_ptr<Variant> ones(xt::xarray<size_t>& shape_array, DType dtype) {
	auto result = std::make_shared<xtv::Variant>();

	DTypeSwitch(dtype, xt::ones, shape_array);

	return result;
}

template <typename operation>
struct BinOperation {
	template<typename A, typename B>
	std::shared_ptr<Variant> operator()(xt::xarray<A>& a, xt::xarray<B>& b) const {
		// ResultType = what results from the usual C++ common promotion of a + b.
		using ResultType = typename std::common_type<A, B>::type;

		// General note: By creating the object first, and assigning later,
		//  we avoid creating the result on the stack first and copying to the heap later.
		// This means this kind of ugly contraption is quite a lot faster than the alternative.
		auto result = std::make_shared<Variant>(xt::xarray<ResultType>());
		
		// Run the operation itself.
		std::get<xt::xarray<ResultType>>(*result) = operation()(a, b);

		// Assign to the result array.
		return result;
	}
};

template <typename operation>
static inline std::shared_ptr<Variant> binOp(Variant& a, Variant& b) {
	return std::visit(BinOperation<operation>{}, a, b);
}

}

#endif
