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

void va::sin(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::sin,
		promote::num_function_result_in_same_out<xt::math::sin_fun>
	>(
		va::XFunction<xt::math::sin_fun> {},
		allocator,
		target,
		array
	);
}

void va::cos(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::cos,
		promote::num_function_result_in_same_out<xt::math::cos_fun>
	>(
		va::XFunction<xt::math::cos_fun> {},
		allocator,
		target,
		array
	);
}

void va::tan(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::tan,
		promote::num_function_result_in_same_out<xt::math::tan_fun>
	>(
		va::XFunction<xt::math::tan_fun> {},
		allocator,
		target,
		array
	);
}

void va::asin(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::asin,
		promote::num_function_result_in_same_out<xt::math::asin_fun>
	>(
		va::XFunction<xt::math::asin_fun> {},
		allocator,
		target,
		array
	);
}

void va::acos(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::acos,
		promote::num_function_result_in_same_out<xt::math::acos_fun>
	>(
		va::XFunction<xt::math::acos_fun> {},
		allocator,
		target,
		array
	);
}

void va::atan(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::atan,
		promote::num_function_result_in_same_out<xt::math::atan_fun>
	>(
		va::XFunction<xt::math::atan_fun> {},
		allocator,
		target,
		array
	);
}

void va::atan2(VStoreAllocator& allocator, const VArrayTarget& target, const VData& x1, const VData& x2) {
	xoperation_inplace<
		Feature::atan2,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::atan2_fun>>
	>(
		va::XFunction<xt::math::atan2_fun> {},
		allocator,
		target,
		x1,
		x2
	);
}

void va::sinh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::sinh,
		promote::num_function_result_in_same_out<xt::math::sinh_fun>
	>(
		va::XFunction<xt::math::sinh_fun> {},
		allocator,
		target,
		array
	);
}

void va::cosh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::cosh,
		promote::num_function_result_in_same_out<xt::math::cosh_fun>
	>(
		va::XFunction<xt::math::cosh_fun> {},
		allocator,
		target,
		array
	);
}

void va::tanh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::tanh,
		promote::num_function_result_in_same_out<xt::math::tanh_fun>
	>(
		va::XFunction<xt::math::tanh_fun> {},
		allocator,
		target,
		array
	);
}

void va::asinh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::asinh,
		promote::num_function_result_in_same_out<xt::math::asinh_fun>
	>(
		va::XFunction<xt::math::asinh_fun> {},
		allocator,
		target,
		array
	);
}

void va::acosh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::acosh,
		promote::num_function_result_in_same_out<xt::math::acosh_fun>
	>(
		va::XFunction<xt::math::acosh_fun> {},
		allocator,
		target,
		array
	);
}

void va::atanh(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::atanh,
		promote::num_function_result_in_same_out<xt::math::atanh_fun>
	>(
		va::XFunction<xt::math::atanh_fun> {},
		allocator,
		target,
		array
	);
}

void va::angle(VStoreAllocator& allocator, const VArrayTarget& target, const std::shared_ptr<VArray>& array) {
	va::atan2(allocator, target, va::imag(array)->data, va::real(array)->data);
}
