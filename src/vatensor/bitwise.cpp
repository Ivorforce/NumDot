#include "bitwise.hpp"

#include <utility>                                      // for move
#include "scalar_tricks.hpp"
#include "varray.hpp"                            // for VArray, VArra...
#include "vconfig.hpp"
#include "vcompute.hpp"                           // for XFunction
#include "vpromote.hpp"                                   // for bool_in_bool_out
#include "xtensor/xlayout.hpp"                          // for layout_type
#include "xtensor/xoperation.hpp"                       // for logical_and

using namespace va;

template <typename A, typename B>
void bitwise_and(VStoreAllocator& allocator, const VArrayTarget& target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::bitwise_and,
		promote::common_int_in_same_out
	>(
		va::XFunction<xt::detail::bitwise_and> {},
		allocator,
		target,
		a,
		b
	);
}

void va::bitwise_and(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::bitwise_and, allocator, target, a, b);
#endif

	::bitwise_and(allocator, target, a, b);
}

template <typename A, typename B>
void bitwise_or(VStoreAllocator& allocator, const VArrayTarget& target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::bitwise_or,
		promote::common_int_in_same_out
	>(
		va::XFunction<xt::detail::bitwise_or> {},
		allocator,
		target,
		a,
		b
	);
}

void va::bitwise_or(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::bitwise_or, allocator, target, a, b);
#endif

	::bitwise_or(allocator, target, a, b);
}

template <typename A, typename B>
void bitwise_xor(VStoreAllocator& allocator, const VArrayTarget& target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::bitwise_xor,
		promote::common_int_in_same_out
	>(
		va::XFunction<xt::detail::bitwise_xor> {},
		allocator,
		target,
		a,
		b
	);
}

void va::bitwise_xor(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_COMMUTATIVE(::bitwise_xor, allocator, target, a, b);
#endif

	::bitwise_xor(allocator, target, a, b);
}

void va::bitwise_not(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a) {
	va::xoperation_inplace<
		Feature::bitwise_not,
		promote::common_int_in_same_out
	>(
		XFunction<xt::detail::bitwise_not> {},
		allocator,
		target,
		a
	);
}

template <typename A, typename B>
void bitwise_left_shift(VStoreAllocator& allocator, const VArrayTarget& target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::bitwise_left_shift,
		promote::left_of_ints_in_same_out
	>(
		va::XFunction<xt::detail::left_shift> {},
		allocator,
		target,
		a,
		b
	);
}

void va::bitwise_left_shift(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_NONCOMMUTATIVE(::bitwise_left_shift, allocator, target, a, b);
#endif

	::bitwise_left_shift(allocator, target, a, b);
}

template <typename A, typename B>
void bitwise_right_shift(VStoreAllocator& allocator, const VArrayTarget& target, const A& a, const B& b) {
	va::xoperation_inplace<
		Feature::bitwise_right_shift,
		promote::left_of_ints_in_same_out
	>(
		va::XFunction<xt::detail::right_shift> {},
		allocator,
		target,
		a,
		b
	);
}

void va::bitwise_right_shift(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
#ifndef NUMDOT_DISABLE_SCALAR_OPTIMIZATION
	OPTIMIZE_NONCOMMUTATIVE(::bitwise_right_shift, allocator, target, a, b);
#endif

	::bitwise_right_shift(allocator, target, a, b);
}
