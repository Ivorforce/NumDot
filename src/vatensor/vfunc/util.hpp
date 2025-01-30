#ifndef VATENSOR_UFUNC_UTIL_HPP
#define VATENSOR_UFUNC_UTIL_HPP

#include <vatensor/vcall.hpp>
#include "tables.hpp"

#define DEFINE_VFUNC_CALLER_UNARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a) {\
	va::call_vfunc_unary(allocator, vfunc::tables::UFUNC_NAME, target, a);\
}

#define DEFINE_RFUNC_CALLER_UNARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const va::axes_type* axes) {\
	va::call_rfunc_unary(allocator, vfunc::tables::UFUNC_NAME, target, a, axes);\
}

#define DEFINE_VFUNC_CALLER_BINARY0(UFUNC_NAME)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {\
	va::call_vfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b);\
}

#define DEFINE_UFUNC_CALLER_BINARY3(UFUNC_NAME, VAR1, VAR2, VAR3)\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, const VAR1 v1, const VAR2 v2, const VAR3 v3) {\
	va::call_vfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b, v1, v2, v3);\
}

#endif //VATENSOR_UFUNC_UTIL_HPP
