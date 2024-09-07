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

using DTypeVariant = std::variant<
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

static DTypeVariant dtype_to_variant(DType dtype) {
	switch (dtype) {
		case xtv::DType::Float32:
			return float_t();
		case xtv::DType::Float64:
			return double_t();
		case xtv::DType::Int8:
			return int8_t();
		case xtv::DType::Int16:
			return int16_t();
		case xtv::DType::Int32:
			return int32_t();
		case xtv::DType::Int64:
			return int64_t();
		case xtv::DType::UInt8:
			return uint8_t();
		case xtv::DType::UInt16:
			return uint16_t();
		case xtv::DType::UInt32:
			return uint32_t();
		case xtv::DType::UInt64:
			return int64_t();
		case xtv::DType::DTypeMax:
			throw std::runtime_error("Invalid dtype.");
	}
}

static inline size_t size_of_dtype_in_bytes(DType dtype) {
	return std::visit([](auto dtype){
		return sizeof(dtype);
	}, dtype_to_variant(dtype));
}

static inline size_t size_of_array_in_bytes(XTVariant& array) {
	return std::visit([](auto& array){
        using V = typename std::decay_t<decltype(array)>::value_type;
		return array.size() * sizeof(V);
	}, array);
}

std::shared_ptr<XTVariant> make_xarray(DType dtype, XTVariant& other) {
	return std::visit([](auto t, auto& other) {
		using T = decltype(t);

		return std::make_shared<XTVariant>( 
			xt::xarray<T>(other)
		);
	}, dtype_to_variant(dtype), other);
}

static std::shared_ptr<XTVariant> get_slice(XTVariant& array, xt::xstrided_slice_vector& slice) {
	return std::visit([slice](auto& array) {
		using T = std::decay_t<decltype(array)>;

		auto view = xt::strided_view(array, slice);

		return std::make_shared<XTVariant>(T(view));
	}, array);
}

template <typename V>
static void set_slice_value(XTVariant& array, xt::xstrided_slice_vector& slice, V value) {
	return std::visit([slice, value](auto& array) {
		xt::strided_view(array, slice) = value;
	}, array);
}

static void set_slice(XTVariant& array, xt::xstrided_slice_vector& slice, XTVariant& value) {
	return std::visit([slice](auto& array, auto& value) {
		xt::strided_view(array, slice) = value;
	}, array, value);
}

template <typename V, typename Sh>
std::shared_ptr<XTVariant> full(const DType dtype, const V fill_value, Sh& shape) {
	return std::visit([fill_value, shape](auto t) {
		using T = decltype(t);
		
		// xt::ones / xt::zeros are slow, I think. Gotta test again though.
		auto ptr = std::make_shared<XTVariant>(xt::xarray<T>::from_shape(shape));
		std::get<xt::xarray<T>>(*ptr).fill(fill_value);
		return ptr;
	}, dtype_to_variant(dtype));
}

template <typename op>
struct XVariantFunction {
// TODO Bind to scons option
#ifndef NUMDOT_ALLOW_MIXED_TYPE_OPS
	// This version exists to reduce the number of functions generated, from O(op n n) to O(o n).
	// Essentially, we make sure that all calls to the function are performed with all types being the same.
	// Every type that doesn't fit will be promoted before the function call.

	// You may think this unnecessary, but it actually reduces the binary size by a factor of 2-3.
	// All 'good' cases, where a promotion is not necessary, retain the same speed.

	template<typename A, typename B>
	std::shared_ptr<XTVariant> operator()(xt::xarray<A>& a, xt::xarray<B>& b) const {
		using ResultType = decltype(std::declval<op>()(std::declval<A>(), std::declval<B>()));
		
		// Note: When making a copy instead of just casting, we can save even more space and compile time.
		//  But it also comes at a larger cost. Just casting as type should be the best of both worlds.
		if constexpr (std::is_same_v<A, B>) {
			// The types are the same, we can just call. If they're wrong, xtensor will promote them for us with optimal performance.
			auto result = op()(a, b);
			return std::make_shared<XTVariant>(xt::xarray<ResultType>(result));
		} else if constexpr (std::is_same_v<A, ResultType>) {
			// a is good, promote b.
			auto result = op()(a, xt::xarray<ResultType>(xt::cast<ResultType>(b)));
			return std::make_shared<XTVariant>(xt::xarray<ResultType>(result));
		} else if constexpr (std::is_same_v<B, ResultType>) {
			// b is good, promote a.
			auto result = op()(xt::cast<ResultType>(a), b);
			return std::make_shared<XTVariant>(xt::xarray<ResultType>(result));
		} else {
			// Both are bad, promote both. This is the worst case, but should be easy to avoid by the programmer if need be.
			auto result = op()(xt::cast<ResultType>(a), xt::cast<ResultType>(b));
			return std::make_shared<XTVariant>(xt::xarray<ResultType>(result));
		}
	}
#endif

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

template <typename T>
static inline T to_single_value(XTVariant& variant) {
	return std::visit([](auto array) {
		if (array.size() != 1) {
			throw std::runtime_error("Array has more than one value; cannot reduce to one.");
		}

		return T(array.flat(0));
	}, variant);
}

}

#endif
