#ifndef NUMDOT_UTIL_HPP
#define NUMDOT_UTIL_HPP

#include <vatensor/vcarray.hpp>
#include "conversion_array.hpp"

namespace numdot {
	template<typename ReturnType, typename Visitor, typename... Args>
	ReturnType reduction(Visitor&& visitor, const Args&... args) {
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
	ReturnType reduction_new(Visitor&& visitor, const Args&... args) {
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

	// Binary reduction with NEP-50 / Array API weak-scalar promotion: if one
	// arg is an NDArray and the other is a Variant scalar (BOOL/INT/FLOAT),
	// the scalar adopts the array's dtype before the visitor runs. Used by
	// ndb/ndf/ndi binary methods (array_equal, all_close, sum_product...).
	template<typename ReturnType, typename Visitor>
	ReturnType reduction_new_binary_weak(Visitor&& visitor, const Variant& a, const Variant& b) {
		try {
			std::shared_ptr<va::VArray> va_arr;
			std::shared_ptr<va::VArray> vb_arr;
			variant_pair_as_arrays_weak(a, b, va_arr, vb_arr);
			ReturnType result = 0;
			va::VData adaptor = va::util::adapt_scalar(&result);
			std::forward<Visitor>(visitor)(&adaptor, *va_arr, *vb_arr);
			return result;
		}
		catch (std::runtime_error& error) {
			ERR_FAIL_V_MSG({}, error.what());
		}
	}
}

#endif //NUMDOT_UTIL_HPP
