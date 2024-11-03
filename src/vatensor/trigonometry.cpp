#include "trigonometry.hpp"

#include <utility>                                       // for move
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for num_function...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for atan2_fun
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

void va::sin(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::sin_fun>>(
		va::XFunction<xt::math::sin_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::cos(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::cos_fun>>(
		va::XFunction<xt::math::cos_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::tan(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::tan_fun>>(
		va::XFunction<xt::math::tan_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::asin(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::asin_fun>>(
		va::XFunction<xt::math::asin_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::acos(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::acos_fun>>(
		va::XFunction<xt::math::acos_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::atan(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::atan_fun>>(
		va::XFunction<xt::math::atan_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::atan2(VStoreAllocator& allocator, VArrayTarget target, const VArray& x1, const VArray& x2) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::atan2_fun>>>(
		va::XFunction<xt::math::atan2_fun> {},
		allocator,
		target,
		x1.data,
		x2.data
	);
#endif
}

void va::sinh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::sinh_fun>>(
		va::XFunction<xt::math::sinh_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::cosh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::cosh_fun>>(
		va::XFunction<xt::math::cosh_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::tanh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::tanh_fun>>(
		va::XFunction<xt::math::tanh_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::asinh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::asinh_fun>>(
		va::XFunction<xt::math::asinh_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::acosh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::acosh_fun>>(
		va::XFunction<xt::math::acosh_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::atanh(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::atanh_fun>>(
		va::XFunction<xt::math::atanh_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}
