#ifndef NUMDOT_XTV_H
#define NUMDOT_XTV_H

#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"

namespace xtv {

using XTVariant = std::variant<
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

using XTVariantContainedTypes = std::tuple<
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
    Float32,
    Float64,
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

static inline DType dtype(XTVariant& variant) {
	return DType(variant.index());
}

static inline xt::svector<size_t> shape(XTVariant& variant) {
	return std::visit([](auto& arg){ return arg.shape(); }, variant);
}

static inline size_t size(XTVariant& variant) {
	return std::visit([](auto& arg){ return arg.size(); }, variant);
}

static inline size_t dimension(XTVariant& variant) {
	return std::visit([](auto& arg){ return arg.dimension(); }, variant);
}

// TODO This should use templates, but i couldn't get it to work.
#define DTypeSwitch(dtype, code, args) switch (dtype) {\
	case xtv::DType::Float32:\
		(*result).emplace<xt::xarray<double_t>>(code<double_t>(args));\
		break;\
	case xtv::DType::Float64:\
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

static std::shared_ptr<XTVariant> array(XTVariant &existing_array, DType dtype) {
    auto result = std::make_shared<XTVariant>();

	// TODO Using the switch here is kinda dumb, but for now it's the easiest way of making it work, making use of std::visit later.
	DTypeSwitch(dtype, xt::xarray, );

	std::visit([](auto& a, auto& b){
		a = b;
	}, *result, existing_array);

    return result;
}

template <typename Sh>
static std::shared_ptr<XTVariant> zeros(Sh& shape_array, DType dtype) {
	// General note: By creating the object first, and assigning later,
	//  we avoid creating the result on the stack first and copying to the heap later.
	// This means this kind of ugly contraption is quite a lot fasterhat than the alternative.
	auto result = std::make_shared<XTVariant>();

	DTypeSwitch(dtype, xt::zeros, shape_array);

	return result;
}

template <typename Sh>
static std::shared_ptr<XTVariant> ones(Sh& shape_array, DType dtype) {
	auto result = std::make_shared<XTVariant>();

	DTypeSwitch(dtype, xt::ones, shape_array);

	return result;
}

template <typename op>
struct BinaryOperation {
	template<typename A, typename B>
	std::shared_ptr<XTVariant> operator()(xt::xarray<A>& a, xt::xarray<B>& b) const {
		// ResultType = what results from the usual C++ common promotion of a + b.
		using ResultType = typename std::common_type<A, B>::type;

		// Note: Need to do this in one line. If the operator is called after the make_shared,
		//  any situations where broadcast errors would be thrown will instead crash the program.
		return std::make_shared<XTVariant>(xt::xarray<ResultType>(op()(a, b)));
	}
};

template <typename op, typename A, typename B>
static inline std::shared_ptr<XTVariant> binary_operation(A& a, B& b) {
	return std::visit(BinaryOperation<op>{}, a, b);
}

// TODO std::add and the likes exist, but it requires type parameters.
//  Therefore, we cannot pass them to operation() as template parameters.
// It would be really nice though if that could be changed though,
//  otherwise we need a good amount of boilerplate for every operation.
struct Add {
	template<typename A, typename B>
	inline auto operator()(A& a, B& b) {
		return a + b;
	}
};

struct Subtract {
	template<typename A, typename B>
	inline auto operator()(A& a, B& b) {
		return a - b;
	}
};

struct Multiply {
	template<typename A, typename B>
	inline auto operator()(A& a, B& b) {
		return a * b;
	}
};

struct Divide {
	template<typename A, typename B>
	inline auto operator()(A& a, B& b) {
		return a / b;
	}
};

}

#endif
