#include "linalg.h"

#include <cstddef>                // for ptrdiff_t
#include <optional>               // for optional
#include <stdexcept>              // for runtime_error
#include <vector>                 // for vector
#include "reduce.h"               // for sum
#include "vatensor/varray.h"      // for VArray, VArrayTarget, VScalar, axes...
#include "vatensor/vassign.h"     // for assign
#include "vmath.h"                // for multiply
#include "xtensor/xslice.hpp"     // for all, ellipsis, newaxis, xall_tag

// struct Dot {
// 	template <typename GivenAxes, typename A, typename B>
// 	auto operator()(GivenAxes&& axes, A&& a, B&& b) const {
// 		auto prod = std::forward<A>(a) * std::forward<B>(b);
// 		return xt::sum(prod, std::forward<GivenAxes>(axes), std::tuple<xt::evaluation_strategy::lazy_type>());
// 	}
//
// 	template <typename A, typename B>
// 	inline auto operator()(A&& a, B&& b) const {
// 		auto prod = std::forward<A>(a) * std::forward<B>(b);
// 		return xt::sum(prod, std::tuple<xt::evaluation_strategy::lazy_type>());
// 	}
// };

va::VScalar va::reduce_dot(const VArray &a, const VArray &b) {
	std::optional<va::VArray> prod_cache;
	va::multiply(&prod_cache, a, b);
	return sum(prod_cache.value());
}

void va::reduce_dot(VArrayTarget target, const VArray &a, const VArray &b, const axes_type& axes) {
	std::optional<va::VArray> prod_cache;
	va::multiply(&prod_cache, a, b);
	va::sum(target, prod_cache.value(), axes);

	// TODO This doesn't work because prod or a and b are lost, either way it crashes.
	// The upside to the above implementation is that no additional code is generated.
	// But it's also a bit slower than if it was fully lazy and accelerated, probably.
	// va::xreduction_inplace<promote::num_matching_float_or_default<double_t>>(
	// 	NormL0{}, axes, target, array.compute_read()
	// );
}

void va::dot(VArrayTarget target, const VArray &a, const VArray &b) {
	if (a.dimension() == 1 && b.dimension() == 1) {
		std::optional<va::VArray> prod_cache;
		va::multiply(&prod_cache, a, b);
		va::assign(target, va::sum(prod_cache.value()));
	}
	else if (a.dimension() == 2 && b.dimension() == 2) {
		return va::matmul(target, a, b);
	}
	else if (a.dimension() == 0 || b.dimension() == 0) {
		return va::multiply(target, a, b);
	}
	else if (b.dimension() == 1) {
		std::optional<va::VArray> prod_cache;
		va::multiply(&prod_cache, a, b);
		va::sum(target, prod_cache.value(), std::vector {static_cast<std::ptrdiff_t>(-1)});
	}
	else {
		throw std::runtime_error("tensordot is not yet implemented");
	}
}

void va::matmul(VArrayTarget target, const VArray &a, const VArray &b) {
	const VArray a_broadcast = a.slice({ xt::ellipsis(), xt::newaxis() });
	const VArray b_broadcast = b.slice({ xt::ellipsis(), xt::newaxis(), xt::all(), xt::all() });

	reduce_dot(target, a_broadcast, b_broadcast, std::vector<std::ptrdiff_t> { -2 });
}
