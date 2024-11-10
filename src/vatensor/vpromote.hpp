#ifndef VPROMOTE_H
#define VPROMOTE_H

#include <complex>
#include <xtensor/xexpression.hpp>
#include <xtensor/xoperation.hpp>
#include <xtl/xcomplex.hpp>

namespace va {
	namespace promote {
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

		template<typename Arg>
		using int64_if_bool_else_id = std::conditional_t<
			std::is_same_v<Arg, bool>,
			int64_t,
			Arg
		>;

		template<typename T>
		inline constexpr bool is_at_least_float_t = std::is_floating_point_v<T> || xtl::is_complex<T>::value;

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

		template<typename NeededType, typename T, std::enable_if_t<!std::is_same_v<NeededType, value_type_v<std::decay_t<T>>>, int> = 0>
		auto promote_value_type_if_needed(const T& arg) {
			if constexpr (std::is_fundamental_v<std::decay_t<T>> || xtl::is_complex<std::decay_t<T>>::value) {
				return static_cast<NeededType>(arg);
			}
			else if constexpr (xtl::is_complex<NeededType>::value) {
				// FIXME Not auto-convertible like below for some reason
				return xt::xarray<NeededType>(xt::cast<NeededType>(arg));
			}
			else {
				// Casting can considerably increase performance (from a small test, it was 25%).
				// However, this is only relevant for operations that even need casting.
				// The cost for casting instead of copying is a much larger binary size (100% increase).
				// Most people will probably prefer the small binary, and accept less optimized wrong dtype operations.
#ifdef NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS
				return xt::cast<NeededType>(std::forward<T>(arg));
#else
				return xt::xarray<NeededType>(arg);
#endif
			}
		}

		template<typename NeededType, typename T, std::enable_if_t<std::is_same_v<NeededType, value_type_v<std::decay_t<T>>>, int> = 0>
		const auto& promote_value_type_if_needed(const T& arg) {
			return arg;
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
			template<typename... Args>
			using input_type = decltype(std::declval<FN>()(std::declval<int64_if_bool_else_id<Args>>()...));

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
		struct num_matching_float_or_default_in_nat_out {
			template<typename... Args>
			using input_type = std::conditional_t<
				is_at_least_float_t<std::common_type_t<int64_if_bool_else_id<Args>...>>,
				std::common_type_t<int64_if_bool_else_id<Args>...>,
				Default
			>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = NaturalOutputType;
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

		struct common_int_in_same_out {
			template<typename... Args>
			using input_type = std::conditional_t<
				std::disjunction_v<std::negation<is_integer_t<Args>>...>,
				void,
				std::common_type_t<Args...>
			>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = NaturalOutputType;
		};

		// e.g. for a << b, a type is more important than b type.
		struct left_of_ints_in_same_out {
			template<typename... Args>
			using input_type = std::conditional_t<
				std::disjunction_v<std::negation<is_integer_t<Args>>...>,
				void,
				std::tuple_element_t<0, std::tuple<Args...>>
			>;

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

		template<typename Default>
		struct num_matching_complex_or_default_in_same_out {
			template <typename Arg, typename Enable = void> struct at_least_float;
			template <typename Arg> struct at_least_float<Arg, std::enable_if_t<xtl::is_complex<Arg>::value>> { using type = typename Arg::value_type; };
			template <typename Arg> struct at_least_float<Arg, std::enable_if_t<std::is_floating_point_v<Arg>>> { using type = Arg; };
			template <typename Arg> struct at_least_float<Arg, std::enable_if_t<!xtl::is_complex<Arg>::value && !std::is_floating_point_v<Arg>>> { using type = Default; };

			template <typename Arg> using at_least_float_v = typename at_least_float<Arg>::type;

			template<typename... Args>
			using input_type = std::complex<std::common_type_t<at_least_float_v<Args>...>>;

			template<typename InputType, typename NaturalOutputType>
			using output_type = InputType;
		};

		template<typename Base>
		struct reject_complex {
			template<bool, typename... Args>
			struct input_type_impl { using type = void; };

			template<typename... Args>
			struct input_type_impl<false, Args...> { using type = typename Base::template input_type<Args...>; };

			template<typename... Args>
			using input_type = typename input_type_impl<std::disjunction_v<xtl::is_complex<std::decay_t<Args>>...>, Args...>::type;

			template<typename InputType, typename NaturalOutputType>
			using output_type = typename Base::template output_type<InputType, NaturalOutputType>;
		};

		template<typename Base>
		struct reject_non_complex {
			template<bool, typename... Args>
			struct input_type_impl { using type = void; };

			template<typename... Args>
			struct input_type_impl<false, Args...> { using type = typename Base::template input_type<Args...>; };

			template<typename... Args>
			using input_type = typename input_type_impl<std::disjunction_v<std::negation<xtl::is_complex<std::decay_t<Args>>...>>, Args...>::type;

			template<typename InputType, typename NaturalOutputType>
			using output_type = typename Base::template output_type<InputType, NaturalOutputType>;
		};
	}
}

#endif //VPROMOTE_H
