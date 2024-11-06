#include "round.hpp"

#include <utility>                                       // for move
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for num_function...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for ceil_fun
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

void va::ceil(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::ceil,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::ceil_fun>>
	>(
		va::XFunction<xt::math::ceil_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::floor(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::floor,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::floor_fun>>
	>(
		va::XFunction<xt::math::floor_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::trunc(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::trunc,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::trunc_fun>>
	>(
		va::XFunction<xt::math::trunc_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::round(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::round,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::round_fun>>
	>(
		va::XFunction<xt::math::round_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::nearbyint(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::nearbyint,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::nearbyint_fun>>\
	>(
		va::XFunction<xt::math::nearbyint_fun> {},
		allocator,
		target,
		array.data
	);
}
