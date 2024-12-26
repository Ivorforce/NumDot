#ifndef VATENSOR_UFUNC_UTIL_HPP
#define VATENSOR_UFUNC_UTIL_HPP

#include <vatensor/vassign.hpp>

#ifdef VA_UFUNC_MODULE
#define IMPLEMENT_UFUNC(CODE)\
namespace VA_UFUNC_MODULE { CODE }
#else
#define IMPLEMENT_UFUNC(CODE)
#endif

#define IMPLEMENT_UNARY_UFUNC(UFUNC_NAME, OP)\
template <typename R, typename A>\
void UFUNC_NAME(R& ret, const A& a) {\
	va::broadcasting_assign(ret, OP);\
}

#define IMPLEMENT_BINARY_UFUNC(UFUNC_NAME, OP)\
template <typename R, typename A, typename B>\
void UFUNC_NAME(R& ret, const A& a, const B& b) {\
	va::broadcasting_assign(ret, OP);\
}

#define UNARY_TABLES(UFUNC_NAME)\
namespace va::ufunc::tables { extern UFuncTableUnary UFUNC_NAME; }

#define UNARY_UFUNC(UFUNC_NAME, OP)\
IMPLEMENT_UFUNC(IMPLEMENT_UNARY_UFUNC(UFUNC_NAME, OP);)\
UNARY_TABLES(UFUNC_NAME)\
namespace va {\
	inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a) {\
		call_ufunc_unary(allocator, ufunc::tables::UFUNC_NAME, target, a);\
	}\
}

#define BINARY_UFUNC(UFUNC_NAME, OP)\
IMPLEMENT_UFUNC(IMPLEMENT_BINARY_UFUNC(UFUNC_NAME, OP);)

#define BINARY_TABLES(UFUNC_NAME)\
namespace va::ufunc::tables { extern UFuncTableBinary UFUNC_NAME; extern UFuncTableBinary UFUNC_NAME##_scalarRight; extern UFuncTableBinary UFUNC_NAME##_scalarLeft; }

#define BINARY_CALLER(UFUNC_NAME)\
BINARY_TABLES(UFUNC_NAME)\
namespace va {\
	inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {\
		if (va::dimension(a) == 0) return call_ufunc_binary(allocator, ufunc::tables::UFUNC_NAME##_scalarLeft, target, va::to_single_value(a), b);\
		if (va::dimension(b) == 0) return call_ufunc_binary(allocator, ufunc::tables::UFUNC_NAME##_scalarRight, target, a, va::to_single_value(b));\
		call_ufunc_binary(allocator, ufunc::tables::UFUNC_NAME, target, a, b);\
	}\
}

#define BINARY_TABLES_COMMUTATIVE(UFUNC_NAME)\
namespace va::ufunc::tables { extern UFuncTableBinary UFUNC_NAME; extern UFuncTableBinary UFUNC_NAME##_scalarRight; }

#define BINARY_CALLER_COMMUTATIVE(UFUNC_NAME)\
BINARY_TABLES_COMMUTATIVE(UFUNC_NAME)\
namespace va {\
	inline void UFUNC_NAME(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {\
		if (va::dimension(a) == 0) return call_ufunc_binary(allocator, ufunc::tables::UFUNC_NAME##_scalarRight, target, b, va::to_single_value(a));\
		if (va::dimension(b) == 0) return call_ufunc_binary(allocator, ufunc::tables::UFUNC_NAME##_scalarRight, target, a, va::to_single_value(b));\
		call_ufunc_binary(allocator, ufunc::tables::UFUNC_NAME, target, a, b);\
	}\
}

#endif //VATENSOR_UFUNC_UTIL_HPP
