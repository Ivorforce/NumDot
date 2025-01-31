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

#ifndef VA_TABLES_EXTERN
#define VA_TABLES_EXTERN extern
#endif

// ReSharper disable CppNonInlineVariableDefinitionInHeaderFile
namespace va::vfunc::tables {
	VA_TABLES_EXTERN UFuncTableUnary negative;
	VA_TABLES_EXTERN UFuncTableUnary sign;
	VA_TABLES_EXTERN UFuncTableUnary abs;
	VA_TABLES_EXTERN UFuncTableUnary square;
	VA_TABLES_EXTERN UFuncTableUnary sqrt;
	VA_TABLES_EXTERN UFuncTableUnary exp;
	VA_TABLES_EXTERN UFuncTableUnary log;
	VA_TABLES_EXTERN UFuncTableUnary rad2deg;
	VA_TABLES_EXTERN UFuncTableUnary deg2rad;

	VA_TABLES_EXTERN UFuncTableUnary conjugate;

	VA_TABLES_EXTERN UFuncTablesBinaryCommutative add;
	VA_TABLES_EXTERN UFuncTablesBinary subtract;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative multiply;
	VA_TABLES_EXTERN UFuncTablesBinary divide;
	VA_TABLES_EXTERN UFuncTablesBinary remainder;
	VA_TABLES_EXTERN UFuncTablesBinary pow;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative minimum;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative maximum;
	
	VA_TABLES_EXTERN UFuncTableUnary sum;
	VA_TABLES_EXTERN UFuncTableUnary prod;
	VA_TABLES_EXTERN UFuncTableUnary mean;
	VA_TABLES_EXTERN UFuncTableUnary variance;
	VA_TABLES_EXTERN UFuncTableUnary standard_deviation;
	VA_TABLES_EXTERN UFuncTableUnary max;
	VA_TABLES_EXTERN UFuncTableUnary min;
	VA_TABLES_EXTERN UFuncTableUnary norm_l0;
	VA_TABLES_EXTERN UFuncTableUnary norm_l1;
	VA_TABLES_EXTERN UFuncTableUnary norm_l2;
	VA_TABLES_EXTERN UFuncTableUnary norm_linf;

	VA_TABLES_EXTERN UFuncTableUnary all;
	VA_TABLES_EXTERN UFuncTableUnary any;

	VA_TABLES_EXTERN UFuncTableUnary sin;
	VA_TABLES_EXTERN UFuncTableUnary cos;
	VA_TABLES_EXTERN UFuncTableUnary tan;
	VA_TABLES_EXTERN UFuncTableUnary asin;
	VA_TABLES_EXTERN UFuncTableUnary acos;
	VA_TABLES_EXTERN UFuncTableUnary atan;
	VA_TABLES_EXTERN UFuncTablesBinary atan2;
	VA_TABLES_EXTERN UFuncTableUnary sinh;
	VA_TABLES_EXTERN UFuncTableUnary cosh;
	VA_TABLES_EXTERN UFuncTableUnary tanh;
	VA_TABLES_EXTERN UFuncTableUnary asinh;
	VA_TABLES_EXTERN UFuncTableUnary acosh;
	VA_TABLES_EXTERN UFuncTableUnary atanh;

	VA_TABLES_EXTERN UFuncTableUnary ceil;
	VA_TABLES_EXTERN UFuncTableUnary floor;
	VA_TABLES_EXTERN UFuncTableUnary trunc;
	VA_TABLES_EXTERN UFuncTableUnary round;
	VA_TABLES_EXTERN UFuncTableUnary rint;

	VA_TABLES_EXTERN UFuncTableUnary logical_not;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative logical_and;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative logical_or;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative logical_xor;

	VA_TABLES_EXTERN UFuncTableUnary bitwise_not;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative bitwise_and;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative bitwise_or;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative bitwise_xor;
	VA_TABLES_EXTERN UFuncTablesBinary bitwise_left_shift;
	VA_TABLES_EXTERN UFuncTablesBinary bitwise_right_shift;

	VA_TABLES_EXTERN UFuncTablesBinaryCommutative equal;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative not_equal;
	VA_TABLES_EXTERN UFuncTablesBinary less;
	VA_TABLES_EXTERN UFuncTablesBinary less_equal;
	VA_TABLES_EXTERN UFuncTablesBinary greater;
	VA_TABLES_EXTERN UFuncTablesBinary greater_equal;

	VA_TABLES_EXTERN UFuncTableUnary isnan;
	VA_TABLES_EXTERN UFuncTableUnary isfinite;
	VA_TABLES_EXTERN UFuncTableUnary isinf;

	VA_TABLES_EXTERN UFuncTablesBinaryCommutative is_close;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative array_equiv;
	VA_TABLES_EXTERN UFuncTablesBinaryCommutative all_close;
}

#endif //TABLES_HPP
