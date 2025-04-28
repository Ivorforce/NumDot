#ifndef VPROMOTE_H
#define VPROMOTE_H

#include <complex>
#include <xtensor/core/xexpression.hpp>
#include <xtensor/core/xoperation.hpp>
#include <xtl/xcomplex.hpp>
#include "create.hpp"
#include "xtensor_store.hpp"

namespace va {
	namespace promote {
		// Type trait to check if T is in std::variant<Args...>
		template<typename T, typename Variant>
		struct is_in_variant;

		template<typename T, typename... Ts>
		struct is_in_variant<T, std::variant<Ts...>> : std::disjunction<std::is_same<T, Ts>...> {};

		// Helper variable template
		template<typename T, typename Variant>
		inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

		template<typename T>
		struct to_64_bit {
			using type = std::conditional_t<std::is_floating_point_v<T>, double_t, int64_t>;
		};

		template<typename T>
		using to_64_bit_t = typename to_64_bit<T>::type;

		template<typename T, typename Variant>
		struct compatible_type_or_64_bit {
			using type = std::conditional_t<is_in_variant_v<T, Variant>, T, to_64_bit_t<T>>;
		};

		template<typename T, typename Variant>
		using compatible_type_or_64_bit_t = typename compatible_type_or_64_bit<T, Variant>::type;

		template <typename T, typename Enable = void>
		struct ValueType;

		// Scalar
		template <typename T>
		struct ValueType<T, std::enable_if_t<std::is_fundamental_v<T>>> {
			using value_type = T;
		};

		// compute case
		template <typename T>
		struct ValueType<T, xt::enable_xexpression<T>> {
			using value_type = typename T::value_type;
		};

		// complex
		template <typename T>
		struct ValueType<T, std::enable_if_t<xtl::is_complex<T>::value>> {
			using value_type = T;
		};

		template<typename T>
		using value_type_v = typename ValueType<T>::value_type;

		template<typename T>
		struct is_integer_t : std::conjunction<
			std::is_integral<T>,
			std::negation<std::is_same<T, bool>>
		> {};

		template<typename T>
		struct is_number_t : std::conjunction<
			std::disjunction<std::is_arithmetic<T>, xtl::is_complex<T>>,
			std::negation<std::is_same<T, bool>>
		> {};

		template <typename T, std::enable_if_t<is_in_variant_v<T, va::VScalar>, int> = 0>
		T deref_data(T&& t) {
			return std::forward<T>(t);
		}

		static const VData& deref_data(const std::shared_ptr<VArray>& t) {
			return t->data;
		}

		static const VData& deref_data(const VData& t) {
			return t;
		}

		static const VScalar& deref_data(const VScalar &t) {
			return t;
		}

		template <typename NeededType>
		NeededType deref_promoted(NeededType&& t) {
			return std::forward<NeededType>(t);
		}

		template <typename NeededType>
		const compute_case<NeededType*>& deref_promoted(const VData& t) {
			return std::get<compute_case<NeededType*>>(t);
		}

		template <typename NeededType>
		NeededType deref_promoted(const VScalar &t) {
			return std::get<NeededType>(t);
		}

		template <typename Need, typename Have, std::enable_if_t<std::is_same_v<std::decay_t<Need>, std::decay_t<Have>>, int> = 0>
		static const Need& promote_list_if_needed(const Have& have) {
			return have;
		}

		template <typename Need, typename Have, std::enable_if_t<!std::is_same_v<std::decay_t<Need>, std::decay_t<Have>>, int> = 0>
		static Need promote_list_if_needed(const Have& have) {
			Need need(have.size());
			std::copy_n(have.begin(), have.size(), need.begin());
			return need;
		}

		template<typename T>
		std::enable_if_t<is_number_t<std::decay_t<T>>::value, T> to_num(T&& b) {
			return std::forward<T>(b);
		}

		// Can't be bool proper because otherwise it will be selected for other primitives by implicit conversion.
		template<typename T>
		std::enable_if_t<std::is_same_v<T, bool>, int64_t> to_num(T b) { return b; }
		template <typename T>
		auto to_num(const xt::xexpression<T>& b) {
			if constexpr (std::is_same_v<value_type_v<T>, bool>) {
				return xt::cast<int64_t>(b.derived_cast());
			}
			else {
				return b.derived_cast();
			}
		}

		template<typename T>
		std::enable_if_t<is_number_t<std::decay_t<T>>::value, bool> to_bool(T&& b) {
			return std::forward<T>(b) != T(0);
		}
		// Can't be bool proper because otherwise it will be selected for other primitives by implicit conversion.
		template<typename T>
		std::enable_if_t<std::is_same_v<T, bool>, bool> to_bool(T b) { return b; }
		template <typename T>
		auto to_bool(const xt::xexpression<T>& b) {
			if constexpr (std::is_same_v<value_type_v<T>, bool>) {
				return b.derived_cast();
			}
			else {
				return xt::not_equal(b.derived_cast(), 0);
			}
		}
	}
}

#endif //VPROMOTE_H
