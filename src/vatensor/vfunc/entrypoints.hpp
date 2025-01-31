#ifndef VATENSOR_UFUNC_CONFIG_HPP
#define VATENSOR_UFUNC_CONFIG_HPP

#include "vatensor/vcall.hpp"
#include "vatensor/vfunc/tables.hpp"

#define DEFINE_VFUNC_CALLER_UNARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a) {\
	va::call_vfunc_unary(allocator, vfunc::tables::UFUNC_NAME, target, a);\
}

#define DEFINE_VFUNC_CALLER_UNARY1(UFUNC_NAME, VAR1)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VAR1 v1) {\
	va::call_vfunc_unary(allocator, vfunc::tables::UFUNC_NAME, target, a, v1);\
}

#define DEFINE_RFUNC_CALLER_UNARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const va::axes_type* axes) {\
	va::call_rfunc_unary(allocator, vfunc::tables::UFUNC_NAME, target, a, axes);\
}

#define DEFINE_VFUNC_CALLER_BINARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {\
	va::call_vfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b);\
}

#define DEFINE_R0FUNC_CALLER_BINARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {\
	va::call_rfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b);\
}

#define DEFINE_RFUNC_CALLER_BINARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, const va::axes_type* axes) {\
	va::call_rfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b, axes);\
}

#define DEFINE_VFUNC_CALLER_BINARY3(UFUNC_NAME, VAR1, VAR2, VAR3)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, const VAR1 v1, const VAR2 v2, const VAR3 v3) {\
	va::call_vfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b, v1, v2, v3);\
}

#define DEFINE_R0FUNC_CALLER_BINARY3(UFUNC_NAME, VAR1, VAR2, VAR3)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, const VAR1 v1, const VAR2 v2, const VAR3 v3) {\
	va::call_rfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b, v1, v2, v3);\
}

namespace va {
	DEFINE_VFUNC_CALLER_UNARY0(negative)
	DEFINE_VFUNC_CALLER_UNARY0(sign)
	DEFINE_VFUNC_CALLER_UNARY0(abs)
	DEFINE_VFUNC_CALLER_UNARY0(square)
	DEFINE_VFUNC_CALLER_UNARY0(sqrt)
	DEFINE_VFUNC_CALLER_UNARY0(exp)
	DEFINE_VFUNC_CALLER_UNARY0(log)
	DEFINE_VFUNC_CALLER_UNARY0(rad2deg)
	DEFINE_VFUNC_CALLER_UNARY0(deg2rad)

	DEFINE_VFUNC_CALLER_UNARY0(conjugate)

	DEFINE_VFUNC_CALLER_BINARY0(add)
	DEFINE_VFUNC_CALLER_BINARY0(subtract)
	DEFINE_VFUNC_CALLER_BINARY0(multiply)
	DEFINE_VFUNC_CALLER_BINARY0(divide)
	DEFINE_VFUNC_CALLER_BINARY0(remainder)
	DEFINE_VFUNC_CALLER_BINARY0(pow)
	DEFINE_VFUNC_CALLER_BINARY0(minimum)
	DEFINE_VFUNC_CALLER_BINARY0(maximum)

	DEFINE_RFUNC_CALLER_UNARY0(sum)
	DEFINE_RFUNC_CALLER_UNARY0(prod)
	DEFINE_RFUNC_CALLER_UNARY0(mean)
	DEFINE_RFUNC_CALLER_UNARY0(median)
	DEFINE_RFUNC_CALLER_UNARY0(variance)
	DEFINE_RFUNC_CALLER_UNARY0(standard_deviation)
	DEFINE_RFUNC_CALLER_UNARY0(max)
	DEFINE_RFUNC_CALLER_UNARY0(min)
	DEFINE_RFUNC_CALLER_UNARY0(norm_l0)
	DEFINE_RFUNC_CALLER_UNARY0(norm_l1)
	DEFINE_RFUNC_CALLER_UNARY0(norm_l2)
	DEFINE_RFUNC_CALLER_UNARY0(norm_linf)

	DEFINE_RFUNC_CALLER_UNARY0(all)
	DEFINE_RFUNC_CALLER_UNARY0(any)

	DEFINE_VFUNC_CALLER_UNARY0(sin)
	DEFINE_VFUNC_CALLER_UNARY0(cos)
	DEFINE_VFUNC_CALLER_UNARY0(tan)
	DEFINE_VFUNC_CALLER_UNARY0(asin)
	DEFINE_VFUNC_CALLER_UNARY0(acos)
	DEFINE_VFUNC_CALLER_UNARY0(atan)
	DEFINE_VFUNC_CALLER_BINARY0(atan2)
	DEFINE_VFUNC_CALLER_UNARY0(sinh)
	DEFINE_VFUNC_CALLER_UNARY0(cosh)
	DEFINE_VFUNC_CALLER_UNARY0(tanh)
	DEFINE_VFUNC_CALLER_UNARY0(asinh)
	DEFINE_VFUNC_CALLER_UNARY0(acosh)
	DEFINE_VFUNC_CALLER_UNARY0(atanh)

	DEFINE_VFUNC_CALLER_UNARY0(ceil)
	DEFINE_VFUNC_CALLER_UNARY0(floor)
	DEFINE_VFUNC_CALLER_UNARY0(trunc)
	DEFINE_VFUNC_CALLER_UNARY0(round)
	DEFINE_VFUNC_CALLER_UNARY0(rint)

	DEFINE_VFUNC_CALLER_UNARY0(logical_not)
	DEFINE_VFUNC_CALLER_BINARY0(logical_and)
	DEFINE_VFUNC_CALLER_BINARY0(logical_or)
	DEFINE_VFUNC_CALLER_BINARY0(logical_xor)

	DEFINE_VFUNC_CALLER_UNARY0(bitwise_not)
	DEFINE_VFUNC_CALLER_BINARY0(bitwise_and)
	DEFINE_VFUNC_CALLER_BINARY0(bitwise_or)
	DEFINE_VFUNC_CALLER_BINARY0(bitwise_xor)
	DEFINE_VFUNC_CALLER_BINARY0(bitwise_left_shift)
	DEFINE_VFUNC_CALLER_BINARY0(bitwise_right_shift)

	DEFINE_VFUNC_CALLER_BINARY0(equal)
	DEFINE_VFUNC_CALLER_BINARY0(not_equal)
	DEFINE_VFUNC_CALLER_BINARY0(less)
	DEFINE_VFUNC_CALLER_BINARY0(less_equal)
	DEFINE_VFUNC_CALLER_BINARY0(greater)
	DEFINE_VFUNC_CALLER_BINARY0(greater_equal)

	DEFINE_VFUNC_CALLER_UNARY0(isnan)
	DEFINE_VFUNC_CALLER_UNARY0(isfinite)
	DEFINE_VFUNC_CALLER_UNARY0(isinf)

	DEFINE_VFUNC_CALLER_BINARY3(is_close, double, double, bool)
	DEFINE_R0FUNC_CALLER_BINARY0(array_equiv)
	static void array_equal(::va::VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
		if (shape(a) != shape(b)) {
			return va::assign(target, false);
		}

		array_equiv(allocator, target, a, b);
	}
	DEFINE_R0FUNC_CALLER_BINARY3(all_close, double, double, bool)

	DEFINE_VFUNC_CALLER_UNARY1(fft, std::ptrdiff_t)

	DEFINE_RFUNC_CALLER_BINARY0(reduce_dot)

	inline void a0xb1_minus_a1xb0(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, const std::ptrdiff_t i0, const std::ptrdiff_t i1) {
		auto a_shape = va::shape(a);
		a_shape.pop_back();
		auto b_shape = va::shape(b);
		b_shape.pop_back();

		const shape_type result_shape = combined_shape(a_shape, b_shape);
		_call_vfunc_binary(allocator, vfunc::tables::a0xb1_minus_a1xb0, target, result_shape, a, b, std::move(i0), std::move(i1));
	}
}

#endif //VATENSOR_UFUNC_CONFIG_HPP
