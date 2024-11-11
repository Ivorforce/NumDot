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

void va::positive(VStoreAllocator& allocator, VArrayTarget target, const VData& a) {
	va::assign(allocator, target, a);
}

void va::negative(VStoreAllocator& allocator, VArrayTarget target, const VData& a) {
	va::xoperation_inplace<
		Feature::negative,
		promote::num_or_error_in_same_out
	>(
		XFunction<xt::detail::negate> {},
		allocator,
		target,
		a
	);
}

#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
void add(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
	va::xoperation_inplace<
		Feature::add,
		promote::num_function_result_in_same_out<xt::detail::plus>
	>(
		XFunction<xt::detail::plus> {},
		allocator,
		target,
		a,
		b
	);
}
#endif

void va::add(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::add, allocator, target, a, b);
#endif

	va::xoperation_inplace<
		Feature::add,
		promote::num_function_result_in_same_out<xt::detail::plus>
	>(
		XFunction<xt::detail::plus> {},
		allocator,
		target,
		a,
		b
	);
}

void va::subtract(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::subtract,
			promote::num_function_result_in_same_out<xt::detail::minus>
		>(
			XFunction<xt::detail::minus> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::subtract,
			promote::num_function_result_in_same_out<xt::detail::minus>
		>(
			XFunction<xt::detail::minus> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

	va::xoperation_inplace<
		Feature::subtract,
		promote::num_function_result_in_same_out<xt::detail::minus>
	>(
		XFunction<xt::detail::minus> {},
		allocator,
		target,
		a,
		b
	);
}

void multiply(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
	va::xoperation_inplace<
		Feature::multiply,
		promote::num_function_result_in_same_out<xt::detail::multiplies>
	>(
		XFunction<xt::detail::multiplies> {},
		allocator,
		target,
		a,
		b
	);
}

void va::multiply(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::multiply, allocator, target, a, b);
#endif

	va::xoperation_inplace<
		Feature::multiply,
		promote::num_function_result_in_same_out<xt::detail::multiplies>
	>(
		XFunction<xt::detail::multiplies> {},
		allocator,
		target,
		a,
		b
	);
}

void va::divide(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::divide,
			promote::num_function_result_in_same_out<xt::detail::divides>
		>(
			XFunction<xt::detail::divides> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::divide,
			promote::num_function_result_in_same_out<xt::detail::divides>
		>(
			XFunction<xt::detail::divides> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

	va::xoperation_inplace<
		Feature::divide,
		promote::num_function_result_in_same_out<xt::detail::divides>
	>(
		XFunction<xt::detail::divides> {},
		allocator,
		target,
		a,
		b
	);
}

void va::remainder(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::remainder,
			promote::reject_complex<promote::num_function_result_in_same_out<xt::math::remainder_fun>>
		>(
			XFunction<xt::math::remainder_fun> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::remainder,
			promote::reject_complex<promote::num_function_result_in_same_out<xt::math::remainder_fun>>
		>(
			XFunction<xt::math::remainder_fun> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

	va::xoperation_inplace<
		Feature::remainder,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::remainder_fun>>
	>(
		XFunction<xt::math::remainder_fun> {},
		allocator,
		target,
		a,
		b
	);
}

void va::pow(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		va::xoperation_inplace<
			Feature::pow,
			promote::num_function_result_in_same_out<xt::math::pow_fun>
		>(
			XFunction<xt::math::pow_fun> {},
			allocator,
			target,
			va::to_single_value(a),
			b
		);
		return;
	}
	if (va::dimension(b) == 0) {
		va::xoperation_inplace<
			Feature::pow,
			promote::num_function_result_in_same_out<xt::math::pow_fun>
		>(
			XFunction<xt::math::pow_fun> {},
			allocator,
			target,
			a,
			va::to_single_value(b)
		);
		return;
	}
#endif

	va::xoperation_inplace<
		Feature::pow,
		promote::num_function_result_in_same_out<xt::math::pow_fun>
	>(
		XFunction<xt::math::pow_fun> {},
		allocator,
		target,
		a,
		b
	);
}

void minimum(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
	va::xoperation_inplace<
		Feature::minimum,
		promote::reject_complex<promote::common_in_same_out>
	>(
		XFunction<xt::math::minimum<void>> {},
		allocator,
		target,
		a,
		b
	);
}

void va::minimum(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::minimum, allocator, target, a, b);
#endif

	va::xoperation_inplace<
		Feature::minimum,
		promote::reject_complex<promote::common_in_same_out>
	>(
		XFunction<xt::math::minimum<void>> {},
		allocator,
		target,
		a,
		b
	);
}

void maximum(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VScalar& b) {
	va::xoperation_inplace<
		Feature::maximum,
		promote::reject_complex<promote::common_in_same_out>
	>(
		XFunction<xt::math::maximum<void>> {},
		allocator,
		target,
		a,
		b
	);
}

void va::maximum(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::maximum, allocator, target, a, b);
#endif

	va::xoperation_inplace<
		Feature::maximum,
		promote::reject_complex<promote::common_in_same_out>
	>(
		XFunction<xt::math::maximum<void>> {},
		allocator,
		target,
		a,
		b
	);
}

void va::clip(VStoreAllocator& allocator, VArrayTarget target, const VData& a, const VData& lo, const VData& hi) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	// TODO Check binary size add and perhaps just use min and max.

	if (va::dimension(lo) == 0 && va::dimension(hi) == 0) {
		va::xoperation_inplace<
			Feature::clip,
			promote::reject_complex<promote::common_in_same_out>
		>(
			XFunction<xt::math::clamp_fun> {},
			allocator,
			target,
			a,
			va::to_single_value(lo),
			va::to_single_value(hi)
		);
		return;
	}
#endif

	va::xoperation_inplace<
		Feature::clip,
		promote::reject_complex<promote::common_in_same_out>
	>(
		XFunction<xt::math::clamp_fun> {},
		allocator,
		target,
		a,
		lo,
		hi
	);
}

void va::sign(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::sign,
		promote::reject_complex<promote::common_in_same_out>
	>(
		va::XFunction<xt::math::sign_fun> {},
		allocator,
		target,
		array
	);
}

void va::abs(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::abs,
		promote::num_in_nat_out
	>(
		va::XFunction<xt::math::abs_fun> {},
		allocator,
		target,
		array
	);
}

// TODO xt::square uses xt::square_fct, which is hidden behind an ifdef.
//  This function is rather harmless, so no idea why, but we can just re-declare it.
struct square_fun {
	template<class T>
	auto operator()(T x) const -> decltype(x * x) {
		return x * x;
	}
};

void va::square(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::square,
		promote::common_in_same_out
	>(
		va::XFunction<square_fun> {},
		allocator,
		target,
		array
	);
}

void va::sqrt(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::sqrt,
		promote::num_function_result_in_same_out<xt::math::sqrt_fun>
	>(
		va::XFunction<xt::math::sqrt_fun> {},
		allocator,
		target,
		array
	);
}

void va::exp(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::exp,
		promote::num_function_result_in_same_out<xt::math::exp_fun>
	>(
		va::XFunction<xt::math::exp_fun> {},
		allocator,
		target,
		array
	);
}

void va::log(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::log,
		promote::num_function_result_in_same_out<xt::math::log_fun>
	>(
		va::XFunction<xt::math::log_fun> {},
		allocator,
		target,
		array
	);
}

void va::rad2deg(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::rad2deg,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::rad2deg>>
	>(
		va::XFunction<xt::math::rad2deg> {},
		allocator,
		target,
		array
	);
}

void va::deg2rad(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	xoperation_inplace<
		Feature::deg2rad,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::deg2rad>>
	>(
		va::XFunction<xt::math::deg2rad> {},
		allocator,
		target,
		array
	);
}

void va::conjugate(VStoreAllocator& allocator, VArrayTarget target, const VData& array) {
	if (!std::visit([](auto x) {
		using VTValue = typename std::decay_t<decltype(x)>::value_type;
		return xtl::is_complex<VTValue>::value;
	}, array)) {
		if (va::dtype(array) == va::Bool) {
			// Need to make sure bools get int-ified.
			va::assign_cast(allocator, target, array, va::Int64);
		}
		else {
			va::assign(allocator, target, array);
		}

		return;
	}

	xoperation_inplace<
		Feature::negative,
		promote::reject_non_complex<promote::common_in_same_out>
	>(
		va::XFunction<xt::math::conj_fun> {},
		allocator,
		target,
		array
	);
}

void va::a0xb1_minus_a1xb0(va::VStoreAllocator& allocator, va::VArrayTarget target, const va::VData& a, const va::VData& b, const std::ptrdiff_t i0, const std::ptrdiff_t i1) {
	va::xoperation_inplace<
		va::Feature::cross,
		va::promote::num_in_same_out
	>(
		[i0, i1](auto& a, auto& b) {
			return (xt::strided_view(a, { xt::ellipsis(), i0 }) * xt::strided_view(b, { xt::ellipsis(), i1 }))
			- (xt::strided_view(a, { xt::ellipsis(), i1 }) * xt::strided_view(b, { xt::ellipsis(), i0 }));
		},
		allocator,
		target,
		a,
		b
	);
}
