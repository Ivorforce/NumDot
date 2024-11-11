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

		cross,

		random_float,
		random_int,
		random_normal,

		fft,
		pad,

		count
	};

	#define FEATURE_NAME_CASE(feature_case) \
	case Feature::feature_case: return #feature_case;

	constexpr const char* feature_name(Feature feature) {
		switch (feature) {
			FEATURE_NAME_CASE(index_masks)
			FEATURE_NAME_CASE(index_lists)

			FEATURE_NAME_CASE(linspace)
			FEATURE_NAME_CASE(arange)

			FEATURE_NAME_CASE(bitwise_and)
			FEATURE_NAME_CASE(bitwise_or)
			FEATURE_NAME_CASE(bitwise_xor)
			FEATURE_NAME_CASE(bitwise_not)
			FEATURE_NAME_CASE(bitwise_left_shift)
			FEATURE_NAME_CASE(bitwise_right_shift)

			FEATURE_NAME_CASE(logical_and)
			FEATURE_NAME_CASE(logical_or)
			FEATURE_NAME_CASE(logical_xor)
			FEATURE_NAME_CASE(logical_not)

			FEATURE_NAME_CASE(equal_to)
			FEATURE_NAME_CASE(not_equal_to)
			FEATURE_NAME_CASE(greater)
			FEATURE_NAME_CASE(greater_equal)
			FEATURE_NAME_CASE(less)
			FEATURE_NAME_CASE(less_equal)

			FEATURE_NAME_CASE(sum)
			FEATURE_NAME_CASE(prod)
			FEATURE_NAME_CASE(mean)
			FEATURE_NAME_CASE(median)
			FEATURE_NAME_CASE(var)
			FEATURE_NAME_CASE(std)
			FEATURE_NAME_CASE(max)
			FEATURE_NAME_CASE(min)
			FEATURE_NAME_CASE(norm_l0)
			FEATURE_NAME_CASE(norm_l1)
			FEATURE_NAME_CASE(norm_l2)
			FEATURE_NAME_CASE(norm_linf)
			FEATURE_NAME_CASE(count_nonzero)
			FEATURE_NAME_CASE(all)
			FEATURE_NAME_CASE(any)
			FEATURE_NAME_CASE(reduce_dot)

			FEATURE_NAME_CASE(ceil)
			FEATURE_NAME_CASE(floor)
			FEATURE_NAME_CASE(trunc)
			FEATURE_NAME_CASE(round)
			FEATURE_NAME_CASE(nearbyint)

			FEATURE_NAME_CASE(sin)
			FEATURE_NAME_CASE(cos)
			FEATURE_NAME_CASE(tan)
			FEATURE_NAME_CASE(asin)
			FEATURE_NAME_CASE(acos)
			FEATURE_NAME_CASE(atan)
			FEATURE_NAME_CASE(atan2)
			FEATURE_NAME_CASE(sinh)
			FEATURE_NAME_CASE(cosh)
			FEATURE_NAME_CASE(tanh)
			FEATURE_NAME_CASE(asinh)
			FEATURE_NAME_CASE(acosh)
			FEATURE_NAME_CASE(atanh)

			FEATURE_NAME_CASE(negative)

			FEATURE_NAME_CASE(add)
			FEATURE_NAME_CASE(subtract)
			FEATURE_NAME_CASE(multiply)
			FEATURE_NAME_CASE(divide)
			FEATURE_NAME_CASE(remainder)
			FEATURE_NAME_CASE(pow)

			FEATURE_NAME_CASE(minimum)
			FEATURE_NAME_CASE(maximum)
			FEATURE_NAME_CASE(clip)

			FEATURE_NAME_CASE(sign)
			FEATURE_NAME_CASE(abs)
			FEATURE_NAME_CASE(square)
			FEATURE_NAME_CASE(sqrt)
			FEATURE_NAME_CASE(exp)
			FEATURE_NAME_CASE(log)

			FEATURE_NAME_CASE(rad2deg)
			FEATURE_NAME_CASE(deg2rad)

			FEATURE_NAME_CASE(random_float)
			FEATURE_NAME_CASE(random_int)
			FEATURE_NAME_CASE(random_normal)

			FEATURE_NAME_CASE(fft)
			FEATURE_NAME_CASE(pad)
			default:
				throw std::invalid_argument("invalid feature");
		}
	}

	#undef FEATURE_NAME_CASE
}

#endif //VFEATURE_HPP
