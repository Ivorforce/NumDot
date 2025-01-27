#ifndef VATENSOR_UFUNC_UTIL_HPP
#define VATENSOR_UFUNC_UTIL_HPP

#include <vatensor/vassign.hpp>

#define UNARY_TABLES(UFUNC_NAME)\
extern UFuncTableUnary UFUNC_NAME;

#define DEFINE_UFUNC_CALLER_UNARY(UFUNC_NAME)\
namespace vfunc::tables { UNARY_TABLES(UFUNC_NAME) }\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a) {\
	call_ufunc_unary(allocator, vfunc::tables::UFUNC_NAME, target, a);\
}

#define BINARY_TABLES(UFUNC_NAME)\
extern UFuncTableBinary UFUNC_NAME; extern UFuncTableBinary UFUNC_NAME##_scalarRight; extern UFuncTableBinary UFUNC_NAME##_scalarLeft;

#define DEFINE_UFUNC_CALLER_BINARY(UFUNC_NAME)\
namespace vfunc::tables { BINARY_TABLES(UFUNC_NAME) }\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {\
	if (va::dimension(a) == 0) return call_ufunc_binary(allocator, vfunc::tables::UFUNC_NAME##_scalarLeft, target, va::to_single_value(a), b);\
	if (va::dimension(b) == 0) return call_ufunc_binary(allocator, vfunc::tables::UFUNC_NAME##_scalarRight, target, a, va::to_single_value(b));\
	call_ufunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b);\
}

#define BINARY_TABLES_COMMUTATIVE(UFUNC_NAME)\
extern UFuncTableBinary UFUNC_NAME; extern UFuncTableBinary UFUNC_NAME##_scalarRight;

#define DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE(UFUNC_NAME)\
namespace vfunc::tables { BINARY_TABLES(UFUNC_NAME) }\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {\
	if (va::dimension(a) == 0) return call_ufunc_binary(allocator, vfunc::tables::UFUNC_NAME##_scalarRight, target, b, va::to_single_value(a));\
	if (va::dimension(b) == 0) return call_ufunc_binary(allocator, vfunc::tables::UFUNC_NAME##_scalarRight, target, a, va::to_single_value(b));\
	call_ufunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b);\
}

#endif //VATENSOR_UFUNC_UTIL_HPP
