#ifndef VFEATURE_HPP
#define VFEATURE_HPP

namespace va {
	enum class Feature {
		index_masks,
		index_lists,

		linspace,
		arange,

		random_float,
		random_int,
		random_normal,

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

			FEATURE_NAME_CASE(random_float)
			FEATURE_NAME_CASE(random_int)
			FEATURE_NAME_CASE(random_normal)

			default:
				throw std::invalid_argument("invalid feature");
		}
	}

	#undef FEATURE_NAME_CASE
}

#endif //VFEATURE_HPP
