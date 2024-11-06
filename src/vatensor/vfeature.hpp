#ifndef VFEATURE_HPP
#define VFEATURE_HPP

namespace va {
	enum class Feature {
		index_masks,
		index_lists,

		linspace,
		arange,

		bitwise_and,
		bitwise_or,
		bitwise_xor,
		bitwise_not,
		bitwise_left_shift,
		bitwise_right_shift,

		logical_and,
		logical_or,
		logical_xor,
		logical_not,

		equal_to,
		not_equal_to,
		greater,
		greater_equal,
		less,
		less_equal,

		sum,
		prod,
		mean,
		median,
		var,
		std,
		max,
		min,
		norm_l0,
		norm_l1,
		norm_l2,
		norm_linf,
		count_nonzero,
		all,
		any,
		reduce_dot,

		ceil,
		floor,
		trunc,
		round,
		nearbyint,

		sin,
		cos,
		tan,
		asin,
		acos,
		atan,
		atan2,
		sinh,
		cosh,
		tanh,
		asinh,
		acosh,
		atanh,

		negative,

		add,
		subtract,
		multiply,
		divide,
		remainder,
		pow,

		minimum,
		maximum,
		clip,

		sign,
		abs,
		square,
		sqrt,
		exp,
		log,

		rad2deg,
		deg2rad,

		random_float,
		random_int,
		random_normal,

		fft,
		pad,

		count
	};
}

#endif //VFEATURE_HPP
