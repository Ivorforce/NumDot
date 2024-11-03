#include "vmath.hpp"

#include <utility>                                          // for move
#include "varray.hpp"                                // for VArray
#include "vcompute.hpp"                               // for XFunction
#include "vpromote.hpp"                                       // for num_funct...
#include "xtensor/xlayout.hpp"                              // for layout_type
#include "xtensor/xmath.hpp"                                // for maximum
#include "xtensor/xoperation.hpp"                           // for divides
#include "xtl/xfunctional.hpp"                              // for select
#include "scalar_tricks.hpp"

using namespace va;

void va::positive(VStoreAllocator& allocator, VArrayTarget target, const VArray& a) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_or_error_in_same_out>(
		XFunction<xt::detail::identity> {},
		allocator,
		target,
		a.data
	);
#endif
}

void va::negative(VStoreAllocator& allocator, VArrayTarget target, const VArray& a) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_or_error_in_same_out>(
		XFunction<xt::detail::negate> {},
		allocator,
		target,
		a.data
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void add(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::plus>>(
		XFunction<xt::detail::plus> {},
		allocator,
		target,
		a.data,
		b
	);
}
#endif

void va::add(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::add, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::plus>>(
		XFunction<xt::detail::plus> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void va::subtract(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::minus>>(
			XFunction<xt::detail::minus> {},
			allocator,
			target,
			a.to_single_value(),
			b.data
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::minus>>(
			XFunction<xt::detail::minus> {},
			allocator,
			target,
			a.data,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::minus>>(
		XFunction<xt::detail::minus> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void multiply(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::multiplies>>(
		XFunction<xt::detail::multiplies> {},
		allocator,
		target,
		a.data,
		b
	);
}

void va::multiply(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::multiply, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::multiplies>>(
		XFunction<xt::detail::multiplies> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void va::divide(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::divides>>(
			XFunction<xt::detail::divides> {},
			allocator,
			target,
			a.to_single_value(),
			b.data
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::divides>>(
			XFunction<xt::detail::divides> {},
			allocator,
			target,
			a.data,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::divides>>(
		XFunction<xt::detail::divides> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void va::remainder(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::remainder_fun>>>(
			XFunction<xt::math::remainder_fun> {},
			allocator,
			target,
			a.to_single_value(),
			b.data
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::remainder_fun>>>(
			XFunction<xt::math::remainder_fun> {},
			allocator,
			target,
			a.data,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::remainder_fun>>>(
		XFunction<xt::math::remainder_fun> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void va::pow(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::math::pow_fun>>(
			XFunction<xt::math::pow_fun> {},
			allocator,
			target,
			a.to_single_value(),
			b.data
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::math::pow_fun>>(
			XFunction<xt::math::pow_fun> {},
			allocator,
			target,
			a.data,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::math::pow_fun>>(
		XFunction<xt::math::pow_fun> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void minimum(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::reject_complex<promote::common_in_same_out>>(
		XFunction<xt::math::minimum<void>> {},
		allocator,
		target,
		a.data,
		b
	);
}

void va::minimum(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::minimum, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::reject_complex<promote::common_in_same_out>>(
		XFunction<xt::math::minimum<void>> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void maximum(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::reject_complex<promote::common_in_same_out>>(
		XFunction<xt::math::maximum<void>> {},
		allocator,
		target,
		a.data,
		b
	);
}

void va::maximum(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::maximum, allocator, target, a, b);
#endif

	va::xoperation_inplace<promote::reject_complex<promote::common_in_same_out>>(
		XFunction<xt::math::maximum<void>> {},
		allocator,
		target,
		a.data,
		b.data
	);
#endif
}

void va::clip(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& lo, const VArray& hi) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	// TODO Check binary size add and perhaps just use min and max.

	if (lo.dimension() == 0 && hi.dimension() == 0) {
		va::xoperation_inplace<promote::reject_complex<promote::common_in_same_out>>(
			XFunction<xt::math::clamp_fun> {},
			allocator,
			target,
			a.data,
			lo.to_single_value(),
			hi.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::reject_complex<promote::common_in_same_out>>(
		XFunction<xt::math::clamp_fun> {},
		allocator,
		target,
		a.data,
		lo.data,
		hi.data
	);
#endif
}

void va::sign(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::common_in_same_out>>(
		va::XFunction<xt::math::sign_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::abs(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_in_nat_out>(
		va::XFunction<xt::math::abs_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

// TODO xt::square uses xt::square_fct, which is hidden behind an ifdef.
//  This function is rather harmless, so no idea why, but we can just re-declare it.
struct square_fun {
	template<class T>
	auto operator()(T x) const -> decltype(x * x) {
		return x * x;
	}
};

void va::square(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::common_in_same_out>(
		va::XFunction<square_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::sqrt(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::sqrt_fun>>(
		va::XFunction<xt::math::sqrt_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::exp(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::exp_fun>>(
		va::XFunction<xt::math::exp_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::log(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::log_fun>>(
		va::XFunction<xt::math::log_fun> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::rad2deg(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::rad2deg>>>(
		va::XFunction<xt::math::rad2deg> {},
		allocator,
		target,
		array.data
	);
#endif
}

void va::deg2rad(VStoreAllocator& allocator, VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::reject_complex<promote::num_function_result_in_same_out<xt::math::deg2rad>>>(
		va::XFunction<xt::math::deg2rad> {},
		allocator,
		target,
		array.data
	);
#endif
}
