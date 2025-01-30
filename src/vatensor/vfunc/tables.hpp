#ifndef TABLES_HPP
#define TABLES_HPP

#include <vatensor/dtype.hpp>

namespace va::vfunc {
	template <int N>
	struct VFunc {
		std::array<va::DType, N> input_types;
		va::DType output_dtype;
		void* function_ptr;
	};
}

namespace va::vfunc::tables {
	using UFuncTableUnary = std::array<VFunc<1>, DTypeMax>;

	using UFuncTableBinary = std::array<std::array<VFunc<2>, DTypeMax>, DTypeMax>;
	struct UFuncTablesBinary {
		UFuncTableBinary tensors;
		UFuncTableBinary scalar_left;
		UFuncTableBinary scalar_right;
	};
	struct UFuncTablesBinaryCommutative {
		UFuncTableBinary tensors;
		UFuncTableBinary scalar_right;
	};
}

#ifndef UNARY_TABLES
#define UNARY_TABLES(UFUNC_NAME) extern UFuncTableUnary UFUNC_NAME;
#endif

#ifndef BINARY_TABLES
#define BINARY_TABLES(UFUNC_NAME) extern UFuncTablesBinary UFUNC_NAME;
#endif

#ifndef BINARY_TABLES_COMMUTATIVE
#define BINARY_TABLES_COMMUTATIVE(UFUNC_NAME) extern UFuncTablesBinaryCommutative UFUNC_NAME;
#endif

// ReSharper disable CppNonInlineVariableDefinitionInHeaderFile
namespace va::vfunc::tables {
	UNARY_TABLES(negative)
	UNARY_TABLES(sign)
	UNARY_TABLES(abs)
	UNARY_TABLES(square)
	UNARY_TABLES(sqrt)
	UNARY_TABLES(exp)
	UNARY_TABLES(log)
	UNARY_TABLES(rad2deg)
	UNARY_TABLES(deg2rad)

	UNARY_TABLES(conjugate)

	BINARY_TABLES_COMMUTATIVE(add)
	BINARY_TABLES(subtract)
	BINARY_TABLES_COMMUTATIVE(multiply)
	BINARY_TABLES(divide)
	BINARY_TABLES(remainder)
	BINARY_TABLES(pow)
	BINARY_TABLES_COMMUTATIVE(minimum)
	BINARY_TABLES_COMMUTATIVE(maximum)

	UNARY_TABLES(sum)

	UNARY_TABLES(sin)
	UNARY_TABLES(cos)
	UNARY_TABLES(tan)
	UNARY_TABLES(asin)
	UNARY_TABLES(acos)
	UNARY_TABLES(atan)
	BINARY_TABLES(atan2);
	UNARY_TABLES(sinh)
	UNARY_TABLES(cosh)
	UNARY_TABLES(tanh)
	UNARY_TABLES(asinh)
	UNARY_TABLES(acosh)
	UNARY_TABLES(atanh)

	UNARY_TABLES(ceil)
	UNARY_TABLES(floor)
	UNARY_TABLES(trunc)
	UNARY_TABLES(round)
	UNARY_TABLES(rint)

	UNARY_TABLES(logical_not)
	BINARY_TABLES_COMMUTATIVE(logical_and);
	BINARY_TABLES_COMMUTATIVE(logical_or);
	BINARY_TABLES_COMMUTATIVE(logical_xor);

	UNARY_TABLES(bitwise_not)
	BINARY_TABLES_COMMUTATIVE(bitwise_and);
	BINARY_TABLES_COMMUTATIVE(bitwise_or);
	BINARY_TABLES_COMMUTATIVE(bitwise_xor);
	BINARY_TABLES(bitwise_left_shift);
	BINARY_TABLES(bitwise_right_shift);

	BINARY_TABLES_COMMUTATIVE(equal);
	BINARY_TABLES_COMMUTATIVE(not_equal);
	BINARY_TABLES(less);
	BINARY_TABLES(less_equal);
	BINARY_TABLES(greater);
	BINARY_TABLES(greater_equal);
	UNARY_TABLES(isnan)
	UNARY_TABLES(isfinite)
	UNARY_TABLES(isinf)

	BINARY_TABLES_COMMUTATIVE(is_close)
}

#endif //TABLES_HPP
