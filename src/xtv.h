#ifndef NUMDOT_XTV_H
#define NUMDOT_XTV_H

#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"
#include "pm_profiler.h"

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

template <typename Func, typename... Args>
static inline auto with_dtype(DType dtype, Args... args) {
	switch (dtype) {
		case xtv::DType::Float32:
			return Func()(float_t(0), args...);
		case xtv::DType::Float64:
			return Func()(double_t(0), args...);
		case xtv::DType::Int8:
			return Func()(int8_t(0), args...);
		case xtv::DType::Int16:
			return Func()(int16_t(0), args...);
		case xtv::DType::Int32:
			return Func()(int32_t(0), args...);
		case xtv::DType::Int64:
			return Func()(int64_t(0), args...);
		case xtv::DType::UInt8:
			return Func()(uint8_t(0), args...);
		case xtv::DType::UInt16:
			return Func()(uint16_t(0), args...);
		case xtv::DType::UInt32:
			return Func()(uint32_t(0), args...);
		case xtv::DType::UInt64:
			return Func()(int64_t(0), args...);
		case xtv::DType::DTypeMax:
			throw std::runtime_error("Invalid dtype.");
	}
}

struct MakeXArray {
	template <typename T, typename O>
	std::shared_ptr<XTVariant> operator()(const T t, const O& other) const {
		return std::make_shared<XTVariant>(xt::xarray<T>(other));
	}
};

static std::shared_ptr<XTVariant> array(XTVariant &existing_array, DType dtype) {
	return std::visit([dtype](auto& existing_array){
		return with_dtype<MakeXArray>(dtype, existing_array);
	}, existing_array);
}

template <int N>
struct Full {
	template <typename T, typename Sh>
	std::shared_ptr<XTVariant> operator()(const T t, Sh&& shape) const {
		// xt::ones / xt::zeros are very slow...
		auto ptr = std::make_shared<XTVariant>(xt::xarray<T>::from_shape(shape));
		std::get<xt::xarray<T>>(*ptr).fill(N);
		return ptr;
	}
};

template <typename op>
struct XVariantFunction {
	template<typename... Args>
	std::shared_ptr<XTVariant> operator()(xt::xarray<Args>&... args) const {
		// ResultType = what results from the native C++ operation op(A(), B())
		using ResultType = decltype(std::declval<op>()(std::declval<Args>()...));

		// TODO We may want to explicitly define promotion types. uint8_t + uint8_t results in an int32, for example.
		// That's for the future though.
		// Also possible:
		// using ResultType = typename std::common_type<Args...>::type;

		// This doesn't do anything yet, it just constructs a value for operation.
		// It will be executed when we use it on the xarray constructor!
		auto result = op()(args...);

		// Note: Need to do this in one line. If the operator is called after the make_shared,
		//  any situations where broadcast errors would be thrown will instead crash the program.
		return std::make_shared<XTVariant>(xt::xarray<ResultType>(result));
	}
};

template<typename FX, typename FN>
struct XFunction {
	// On xarray input: Make the xfunction.
	// This is analogous to xt::add etc., with the main difference that in our setup it's easier to use this function with the
	//  appropriate xt::detail:: operation.
	template<typename... Args>
	inline auto operator()(xt::xarray<Args>&&... args) const -> xt::detail::xfunction_type_t<FX, xt::xarray<Args>...> {
		return xt::detail::make_xfunction<FX>(std::forward<xt::xarray<Args>>(args)...);
	}

	// On normal input: Run the normal function. 
	// This is used by XVariantFunction to infer the dtype of the result.
	template<typename... Args>
	inline auto operator()(Args&&... args) const -> decltype(std::declval<FN>()(std::forward<Args>(args)...)) {
		return FN()(std::forward<Args>(args)...);
	}
};

template <typename FX, typename FN, typename... Args>
static inline std::shared_ptr<XTVariant> xoperation(Args&&... args) {
	return std::visit(XVariantFunction<XFunction<FX, FN>>{}, std::forward<Args>(args)...);
}

}

#endif
