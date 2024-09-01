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
inline std::shared_ptr<Variant> binOp(Variant& a, Variant& b) {
	return std::visit(BinOperation<operation>{}, a, b);
}

}

#endif
