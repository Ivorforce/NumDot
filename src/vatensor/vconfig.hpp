#ifndef NUMDOT_VCONFIG_HPP
#define NUMDOT_VCONFIG_HPP

#include <array>
#include <utility>

#include "vfeature.hpp"

#ifdef NUMDOT_USE_USER_CONFIG
#include "gen/userconfig.hpp"
#endif

namespace va {
	constexpr std::array<bool, static_cast<size_t>(Feature::count)> make_is_enabled_map(std::initializer_list<std::pair<Feature, bool>> init) {
		std::array<bool, static_cast<size_t>(Feature::count)> arr{};
		// Need c++20 to use fill here.
		// arr.fill(true);

		for (std::size_t i = 0; i < static_cast<std::size_t>(Feature::count); ++i) {
			arr[i] = false;
		}

		for (const auto& [key, value] : init) {
			arr[static_cast<std::size_t>(key)] = value;
		}

		return arr;
	}

	// Initialize the array with explicit mappings
	constexpr auto is_enabled_by_feature = make_is_enabled_map(
#ifdef NUMDOT_USE_USER_CONFIG
		va::userconfig::is_enabled_by_feature_initializer
#else
		{}
#endif
	);

	constexpr bool is_feature_enabled(Feature value) {
		return is_enabled_by_feature[static_cast<size_t>(value)];
	}
}

#endif //NUMDOT_VCONFIG_HPP
