#ifndef VPROMOTE_H
#define VPROMOTE_H

#include <complex>
#include <xtensor/xexpression.hpp>
#include <xtensor/xoperation.hpp>

namespace va {
	namespace promote {
		template<typename T>
		struct is_complex_t : std::false_type {};

		template<typename T>
		struct is_complex_t<std::complex<T>> : std::true_type {};

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
		struct ValueType<T, std::enable_if_t<is_complex_t<T>{}>> {
			using value_type = T;
		};

		template<typename T>
		using value_type_v = typename ValueType<T>::value_type;

		template<typename Arg>
		using int64_if_bool_else_id = std::conditional_t<
			std::is_same_v<Arg, bool>,
			int64_t,
			Arg
		>;

		template<typename T>
		inline constexpr bool is_at_least_float_t = std::is_floating_point_v<T> || is_complex_t<T>::value;

		template<typename T>
		struct is_number_t : std::conjunction<
			std::disjunction<std::is_arithmetic<T>, is_complex_t<T>>,
			std::negation<std::is_same<T, bool>>
		> {};

		template<typename NeededType, typename T>
		auto promote_value_type_if_needed(T&& arg) {
			using V = value_type_v<std::decay_t<decltype(arg)>>;

			if constexpr (std::is_same_v<V, NeededType>) {
				// Most common situation: the argument we need is the same as the argument that's given.
				return std::forward<T>(arg);
			}
			else {
				if constexpr (std::is_fundamental_v<T>) {
					return static_cast<NeededType>(std::forward<T>(arg));
				}
				else {
					// Casting can considerably increase performance (from a small test, it was 25%).
					// However, this is only relevant for operations that even need casting.
					// The cost for casting instead of copying is a much larger binary size (100% increase).
					// Most people will probably prefer the small binary, and accept less optimized wrong dtype operations.
#ifdef NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS
			        return xt::cast<NeededType>(std::forward<T>(arg));
#else
					return xt::xarray<NeededType>(std::forward<T>(arg));
#endif
				}
			}
		}

		// TODO We should merge this with the above
		template<typename NeededType, typename T>
		auto promote_value_type_if_needed_fast(T&& arg) {
			using V = value_type_v<std::decay_t<decltype(arg)>>;

			if constexpr (std::is_same_v<V, NeededType>) {
				// Most common situation: the argument we need is the same as the argument that's given.
				return std::forward<T>(arg);
			}
			else {
				if constexpr (std::is_fundamental_v<T>) {
					return static_cast<NeededType>(std::forward<T>(arg));
				}
				else {
					return xt::cast<NeededType>(std::forward<T>(arg));
				}
			}
		}

		// TODO We may want to support mixed-type input ops for some functions, to avoid explicitly promoting types.
		//  I think it may be faster to not cast beforehand, but it's possible it does it later down the line anyway.
		//  That should be tested.
		// Also, mixed-type input ops can really increase binary size, so it should be used with care if at all.

		struct num_in_same_out {
			template<typename... Args>
			using input_type = std::common_type_t<int64_if_bool_else_id<Args>...>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = InputType;
		};

		struct num_or_error_in_same_out {
			template<typename... Args>
			using input_type = std::conditional_t<
				!std::conjunction_v<is_number_t<Args>...>,
				void,
				std::common_type_t<Args...>
			>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = InputType;
		};

		/**
		 * 	what results from the native C++ operation op(A(), B())
		 */
		template<typename FN>
		struct num_function_result_in_same_out {
			template<bool, typename... Args>
			struct input_type_impl { using type = void; };

			template<typename... Args>
			struct input_type_impl<false, Args...> {
				using type = decltype(std::declval<FN>()(std::declval<int64_if_bool_else_id<Args>>()...));
			};

			template<typename... Args>
			using input_type = typename input_type_impl<std::disjunction_v<is_complex_t<std::decay_t<Args>>...>, Args...>::type;

			template<typename InputType, typename NaturalOutputType>
			using output_type = InputType;
		};

		struct num_at_least_int32_in_same_out {
			template<typename... Args>
			using common_type = std::common_type_t<int64_if_bool_else_id<Args>...>;

			template<typename... Args>
			using input_type = std::conditional_t<
				(std::numeric_limits<common_type<Args...>>::digits >= std::numeric_limits<int32_t>::digits),
				common_type<Args...>,
				std::conditional_t<
					std::is_signed_v<common_type<Args...>>,
					int32_t,
					uint32_t
				>
			>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = InputType;
		};

		template<typename Default>
		struct num_matching_float_or_default_in_same_out {
			template<typename... Args>
			using input_type = std::conditional_t<
				is_at_least_float_t<std::common_type_t<int64_if_bool_else_id<Args>...>>,
				std::common_type_t<int64_if_bool_else_id<Args>...>,
				Default
			>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = InputType;
		};

		struct common_in_same_out {
			template<typename... Args>
			using input_type = std::common_type_t<Args...>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = InputType;
		};

		struct num_in_nat_out {
			template<typename... Args>
			using input_type = std::common_type_t<int64_if_bool_else_id<Args>...>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = NaturalOutputType;
		};

		struct common_in_nat_out {
			template<typename... Args>
			using input_type = std::common_type_t<Args...>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = NaturalOutputType;
		};

		template<typename Type>
		struct x_in_nat_out {
			template<typename... Args>
			using input_type = Type;

			template<typename InputType, typename NaturalOutputType>
			using output_type = NaturalOutputType;
		};
	}
}

#endif //VPROMOTE_H
