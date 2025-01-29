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

#define DEFINE_VFUNC_CALLER_BINARY(UFUNC_NAME)\
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

#define DEFINE_UFUNC_CALLER_BINARY_COMMUTATIVE3(UFUNC_NAME, VAR1, VAR2, VAR3)\
namespace vfunc::tables { BINARY_TABLES(UFUNC_NAME) }\
inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, const VAR1 v1, const VAR2 v2, const VAR3 v3) {\
	if (va::dimension(a) == 0) return call_vfunc_binary(allocator, vfunc::tables::UFUNC_NAME##_scalarRight, target, b, va::to_single_value(a), v1, v2, v3);\
	if (va::dimension(b) == 0) return call_vfunc_binary(allocator, vfunc::tables::UFUNC_NAME##_scalarRight, target, a, va::to_single_value(b), v1, v2, v3);\
	call_vfunc_binary(allocator, vfunc::tables::UFUNC_NAME, target, a, b, v1, v2, v3);\
}

#endif //VATENSOR_UFUNC_UTIL_HPP
