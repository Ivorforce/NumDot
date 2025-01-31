#ifndef VFEATURE_HPP
#define VFEATURE_HPP

namespace va {
	enum class Feature {
		index_masks,
		index_lists,

		linspace,
		arange,

		array_equal,
		all_close,

		median,
		var,
		std,
		max,
		min,
		norm_l0,
		norm_l1,
		norm_l2,
		norm_linf,
		all,
		any,
		reduce_dot,

		clip,

		cross,

		random_float,
		random_int,
		random_normal,

		fft,
		pad,

		count,
	};

	#define FEATURE_NAME_CASE(feature_case) \
	case Feature::feature_case: return #feature_case;

	constexpr const char* feature_name(Feature feature) {
		switch (feature) {
			FEATURE_NAME_CASE(index_masks)
			FEATURE_NAME_CASE(index_lists)

			FEATURE_NAME_CASE(linspace)
			FEATURE_NAME_CASE(arange)

			FEATURE_NAME_CASE(array_equal)
			FEATURE_NAME_CASE(all_close)

			FEATURE_NAME_CASE(median)
			FEATURE_NAME_CASE(var)
			FEATURE_NAME_CASE(std)
			FEATURE_NAME_CASE(max)
			FEATURE_NAME_CASE(min)
			FEATURE_NAME_CASE(norm_l0)
			FEATURE_NAME_CASE(norm_l1)
			FEATURE_NAME_CASE(norm_l2)
			FEATURE_NAME_CASE(norm_linf)
			FEATURE_NAME_CASE(all)
			FEATURE_NAME_CASE(any)
			FEATURE_NAME_CASE(reduce_dot)

			FEATURE_NAME_CASE(clip)

			FEATURE_NAME_CASE(cross)

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
