#include "trigonometry.hpp"

#include <utility>                                       // for move

#include "rearrange.hpp"
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for num_function...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for atan2_fun
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

void va::sin(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::sin,
		promote::num_function_result_in_same_out<xt::math::sin_fun>
	>(
		va::XFunction<xt::math::sin_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::cos(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::cos,
		promote::num_function_result_in_same_out<xt::math::cos_fun>
	>(
		va::XFunction<xt::math::cos_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::tan(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::tan,
		promote::num_function_result_in_same_out<xt::math::tan_fun>
	>(
		va::XFunction<xt::math::tan_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::asin(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::asin,
		promote::num_function_result_in_same_out<xt::math::asin_fun>
	>(
		va::XFunction<xt::math::asin_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::acos(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::acos,
		promote::num_function_result_in_same_out<xt::math::acos_fun>
	>(
		va::XFunction<xt::math::acos_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::atan(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::atan,
		promote::num_function_result_in_same_out<xt::math::atan_fun>
	>(
		va::XFunction<xt::math::atan_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::atan2(VStoreAllocator& allocator, VArrayTarget target, const VArray& x1, const VArray& x2) {
	xoperation_inplace<
		Feature::atan2,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::atan2_fun>>
	>(
		va::XFunction<xt::math::atan2_fun> {},
		allocator,
		target,
		x1.data,
		x2.data
	);
}

void va::sinh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::sinh,
		promote::num_function_result_in_same_out<xt::math::sinh_fun>
	>(
		va::XFunction<xt::math::sinh_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::cosh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::cosh,
		promote::num_function_result_in_same_out<xt::math::cosh_fun>
	>(
		va::XFunction<xt::math::cosh_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::tanh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::tanh,
		promote::num_function_result_in_same_out<xt::math::tanh_fun>
	>(
		va::XFunction<xt::math::tanh_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::asinh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::asinh,
		promote::num_function_result_in_same_out<xt::math::asinh_fun>
	>(
		va::XFunction<xt::math::asinh_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::acosh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::acosh,
		promote::num_function_result_in_same_out<xt::math::acosh_fun>
	>(
		va::XFunction<xt::math::acosh_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::atanh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
	xoperation_inplace<
		Feature::atanh,
		promote::num_function_result_in_same_out<xt::math::atanh_fun>
	>(
		va::XFunction<xt::math::atanh_fun> {},
		allocator,
		target,
		array.data
	);
}

void va::angle(VStoreAllocator& allocator, VArrayTarget target, const std::shared_ptr<VArray>& array) {
	va::atan2(allocator, target, *va::imag(array), *va::real(array));
}
