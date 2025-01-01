#include "comparison.hpp"

#include <utility>                                       // for move
#include "scalar_tricks.hpp"
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for common_num_i...
#include "xtensor/xmath.hpp"                           // for layout_type
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xoperation.hpp"                        // for equal_to
// See below
#ifdef _WIN32
#include "ufunc/ufunc_features.hpp"
#include "xtensor_store.hpp"
#include "reduce.hpp"
#endif

using namespace va;

template <typename A, typename B>
void is_close(VStoreAllocator& allocator, const VArrayTarget& target, const A& a, const B& b, double rtol, double atol, bool equal_nan) {
	va::xoperation_inplace<
		Feature::is_close,
		promote::common_in_nat_out
	>(
		[rtol, atol, equal_nan](auto&& a, auto&& b) {
			return xt::isclose(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b), rtol, atol, equal_nan);
		},
		allocator,
		target,
		a,
		b
	);
}

void va::is_close(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, double rtol, double atol, bool equal_nan) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		::is_close(allocator, target, b, va::to_single_value(a), rtol, atol, equal_nan);
		return;
	}
	if (va::dimension(b) == 0) {
		::is_close(allocator, target, a, va::to_single_value(b), rtol, atol, equal_nan);
		return;
	}
#endif

	::is_close(allocator, target, a, b, rtol, atol, equal_nan);
}

bool va::array_equiv(const VData& a, const VData& b) {
#ifndef _WIN32
	try {
		return vreduce<
			Feature::array_equal,
			promote::common_in_nat_out,
			bool
		>(
			[](auto&& a, auto&& b) -> bool {
				return xt::all(xt::equal(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b)));
			},
			a,
			b
		);
	}
	catch (std::runtime_error& e) {
		// Should be broadcast error or cast error.
		return false;
	}
#else
	std::shared_ptr<VArray> intermediate;
	va::equal(va::store::default_allocator, &intermediate, a, b);
	return va::all(intermediate->data);
#endif
}

bool va::array_equal(const VData& a, const VData& b) {
	if (shape(a) != shape(b)) {
		return false;
	}

	return array_equiv(a, b);
}

template <typename A, typename B>
bool all_close(const A& a, const B& b, double rtol, double atol, bool equal_nan) {
	return vreduce<
		Feature::all_close,
		promote::common_in_nat_out,
		bool
	>(
		[rtol, atol, equal_nan](auto&& a, auto&& b) -> bool {
			return xt::all(xt::isclose(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b), rtol, atol, equal_nan));
		},
		a,
		b
	);
}

bool va::all_close(const VData& a, const VData& b, double rtol, double atol, bool equal_nan) {
	// TODO No idea why but the windows compiler refuses to compile this one for some reason,
	//  claiming that vreduce's ReturnType does not exist.
#ifndef _WIN32
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	if (va::dimension(a) == 0) {
		return ::all_close(b, va::to_single_value(a), rtol, atol, equal_nan);
	}
	if (va::dimension(b) == 0) {
		return ::all_close(a, va::to_single_value(b), rtol, atol, equal_nan);
	}
#endif

	return ::all_close(a, b, rtol, atol, equal_nan);
#else
	std::shared_ptr<VArray> intermediate;
	::is_close(va::store::default_allocator, &intermediate, a, b, rtol, atol, equal_nan);
	return va::all(intermediate->data);
#endif
}
