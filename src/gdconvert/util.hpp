#ifndef NUMDOT_UTIL_HPP
#define NUMDOT_UTIL_HPP

#include <vatensor/vcarray.hpp>

namespace numdot {
	template<typename ReturnType, typename Visitor, typename... Args>
	static ReturnType reduction(Visitor&& visitor, const Args&... args) {
		try {
			const auto result = std::forward<Visitor>(visitor)(*variant_as_array(args)...);

			if constexpr (std::is_same_v<std::decay_t<decltype(result)>, va::VScalar>) {
				return va::static_cast_scalar<ReturnType>(result);
			}
			else {
				return result;
			}
		}
		catch (std::runtime_error& error) {
			ERR_FAIL_V_MSG({}, error.what());
		}
	}

	template<typename ReturnType, typename Visitor, typename... Args>
	static ReturnType reduction_new(Visitor&& visitor, const Args&... args) {
		try {
			ReturnType result = 0;
			va::VData adaptor = va::util::adapt_scalar(&result);
			std::forward<Visitor>(visitor)(&adaptor, *variant_as_array(args)...);
			return result;
		}
		catch (std::runtime_error& error) {
			ERR_FAIL_V_MSG({}, error.what());
		}
	}
}

#endif //NUMDOT_UTIL_HPP
