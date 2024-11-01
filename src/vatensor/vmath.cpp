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

void va::positive(VArrayTarget target, const VArray& a) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_or_error_in_same_out>(
		XFunction<xt::detail::identity> {},
		target,
		a.read
	);
#endif
}

void va::negative(VArrayTarget target, const VArray& a) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	va::xoperation_inplace<promote::num_or_error_in_same_out>(
		XFunction<xt::detail::negate> {},
		target,
		a.read
	);
#endif
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void add(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::plus>>(
		XFunction<xt::detail::plus> {},
		target,
		a.read,
		b
	);
}
#endif

void va::add(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::add, a, b);
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::plus>>(
		XFunction<xt::detail::plus> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::subtract(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::minus>>(
			XFunction<xt::detail::minus> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::minus>>(
			XFunction<xt::detail::minus> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::minus>>(
		XFunction<xt::detail::minus> {},
		target,
		a.read,
		b.read
	);
#endif
}

void multiply(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::multiplies>>(
		XFunction<xt::detail::multiplies> {},
		target,
		a.read,
		b
	);
}

void va::multiply(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::multiply, a, b);
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::multiplies>>(
		XFunction<xt::detail::multiplies> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::divide(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::divides>>(
			XFunction<xt::detail::divides> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::divides>>(
			XFunction<xt::detail::divides> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::detail::divides>>(
		XFunction<xt::detail::divides> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::remainder(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out_no_complex<xt::math::remainder_fun>>(
			XFunction<xt::math::remainder_fun> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out_no_complex<xt::math::remainder_fun>>(
			XFunction<xt::math::remainder_fun> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out_no_complex<xt::math::remainder_fun>>(
		XFunction<xt::math::remainder_fun> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::pow(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (a.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::math::pow_fun>>(
			XFunction<xt::math::pow_fun> {},
			target,
			a.to_single_value(),
			b.read
		);
		return;
	}
	if (b.dimension() == 0) {
		va::xoperation_inplace<promote::num_function_result_in_same_out<xt::math::pow_fun>>(
			XFunction<xt::math::pow_fun> {},
			target,
			a.read,
			b.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::num_function_result_in_same_out<xt::math::pow_fun>>(
		XFunction<xt::math::pow_fun> {},
		target,
		a.read,
		b.read
	);
#endif
}

void minimum(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::common_in_same_out>(
		XFunction<xt::math::minimum<void>> {},
		target,
		a.read,
		b
	);
}

void va::minimum(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::minimum, a, b);
#endif

	va::xoperation_inplace<promote::common_in_same_out>(
		XFunction<xt::math::minimum<void>> {},
		target,
		a.read,
		b.read
	);
#endif
}

void maximum(VArrayTarget target, const VArray& a, const VScalar& b) {
	va::xoperation_inplace<promote::common_in_same_out>(
		XFunction<xt::math::maximum<void>> {},
		target,
		a.read,
		b
	);
}

void va::maximum(VArrayTarget target, const VArray& a, const VArray& b) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::maximum, a, b);
#endif

	va::xoperation_inplace<promote::common_in_same_out>(
		XFunction<xt::math::maximum<void>> {},
		target,
		a.read,
		b.read
	);
#endif
}

void va::clip(VArrayTarget target, const VArray& a, const VArray& lo, const VArray& hi) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (lo.dimension() == 0 && hi.dimension() == 0) {
		va::xoperation_inplace<promote::common_in_same_out>(
			XFunction<xt::math::clamp_fun> {},
			target,
			a.read,
			lo.to_single_value(),
			hi.to_single_value()
		);
		return;
	}
#endif

	va::xoperation_inplace<promote::common_in_same_out>(
		XFunction<xt::math::clamp_fun> {},
		target,
		a.read,
		lo.read,
		hi.read
	);
#endif
}

void va::sign(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_or_error_in_same_out>(
		va::XFunction<xt::math::sign_fun> {},
		target,
		array.read
	);
#endif
}

void va::abs(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::abs_fun>>(
		va::XFunction<xt::math::abs_fun> {},
		target,
		array.read
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

void va::square(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::common_in_same_out>(
		va::XFunction<square_fun> {},
		target,
		array.read
	);
#endif
}

void va::sqrt(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::sqrt_fun>>(
		va::XFunction<xt::math::sqrt_fun> {},
		target,
		array.read
	);
#endif
}

void va::exp(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::exp_fun>>(
		va::XFunction<xt::math::exp_fun> {},
		target,
		array.read
	);
#endif
}

void va::log(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out<xt::math::log_fun>>(
		va::XFunction<xt::math::log_fun> {},
		target,
		array.read
	);
#endif
}

void va::rad2deg(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out_no_complex<xt::math::rad2deg>>(
		va::XFunction<xt::math::rad2deg> {},
		target,
		array.read
	);
#endif
}

void va::deg2rad(VArrayTarget target, const VArray& array) {
#ifdef NUMDOT_DISABLE_MATH_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_MATH_FUNCTIONS to enable it.");
#else
	xoperation_inplace<promote::num_function_result_in_same_out_no_complex<xt::math::deg2rad>>(
		va::XFunction<xt::math::deg2rad> {},
		target,
		array.read
	);
#endif
}
