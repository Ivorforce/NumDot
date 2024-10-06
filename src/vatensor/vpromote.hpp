#ifndef VPROMOTE_H
#define VPROMOTE_H

namespace va {
	namespace promote {
		template <typename T, typename Enable = void>
		struct ValueType;

		template <typename T>
		struct ValueType<T, std::enable_if_t<std::is_fundamental_v<T>>> {
			using value_type = T;
		};

		template <typename T>
		struct ValueType<T, std::enable_if_t<!std::is_fundamental_v<T>>> {
			using value_type = typename T::value_type;
		};

		template<typename T>
		using value_type_v = typename ValueType<T>::value_type;

		template<typename NeededType, typename T>
		auto promote_value_type_if_needed(T&& arg) {
			using V = value_type_v<std::decay_t<decltype(arg)>>;

			if constexpr (std::is_same_v<V, NeededType>) {
				// Most common situation: the argument we need is the same as the argument that's given.
				return std::forward<T>(arg);
			}
			else {
				if constexpr (std::is_fundamental_v<T>) {
					return static_cast<NeededType>(arg);
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

		template<typename Arg>
		using int64_if_bool_else_id = std::conditional_t<
			std::is_same_v<Arg, bool>,
			int64_t,
			Arg
		>;

		template<typename T>
		struct is_non_bool_arithmetic : std::conjunction<std::is_arithmetic<T>, std::negation<std::is_same<T, bool>>> {};

		// TODO We may want to support mixed-type input ops for some functions, to avoid explicitly promoting types.
		//  I think it may be faster to not cast beforehand, but it's possible it does it later down the line anyway.
		//  That should be tested.
		// Also, mixed-type input ops can really increase binary size, so it should be used with care if at all.

		template<typename OutputType>
		struct common_num_in_x_out {
			template<typename... Args>
			using input_type = std::common_type_t<int64_if_bool_else_id<Args>...>;

			template<typename InputType>
			using output_type = OutputType;
		};

		struct common_num_or_error {
			template<typename... Args>
			using input_type = std::conditional_t<
				!std::conjunction_v<is_non_bool_arithmetic<Args>...>,
				void,
				std::common_type_t<Args...>
			>;

			template<typename InputType>
			using output_type = InputType;
		};

		/**
		 * 	what results from the native C++ operation op(A(), B())
		 */
		template<typename FN>
		struct num_function_result {
			template<typename... Args>
			using input_type = decltype(std::declval<FN>()(std::declval<int64_if_bool_else_id<Args>>()...));

			template<typename InputType>
			using output_type = InputType;
		};

		struct num_common_type {
			template<typename... Args>
			using input_type = std::common_type_t<int64_if_bool_else_id<Args>...>;

			template<typename InputType>
			using output_type = InputType;
		};

		struct num_common_at_least_int32 {
			template<typename... Args>
			using common_type = std::common_type_t<int64_if_bool_else_id<Args>...>;

			template<typename... Args>
			using input_type = typename std::conditional<
				(std::numeric_limits<common_type<Args...>>::digits >= std::numeric_limits<int32_t>::digits),
				common_type<Args...>,
				typename std::conditional<
					std::is_signed<common_type<Args...>>::value,
					int32_t,
					uint32_t
				>::type
			>::type;

			template<typename InputType>
			using output_type = InputType;
		};

		template<typename Default>
		struct num_matching_float_or_default {
			template<typename... Args>
			using input_type = std::conditional_t<
				std::is_floating_point_v<std::common_type_t<int64_if_bool_else_id<Args>...>>,
				std::common_type_t<int64_if_bool_else_id<Args>...>,
				Default
			>;

			template<typename InputType>
			using output_type = InputType;
		};

		struct common_in_common_out {
			template<typename... Args>
			using input_type = std::common_type_t<Args...>;

			template<typename InputType>
			using output_type = InputType;
		};

		struct common_in_bool_out {
			template<typename... Args>
			using input_type = std::common_type_t<Args...>;

			template<typename InputType>
			using output_type = bool;
		};

		struct bool_in_bool_out {
			template<typename... Args>
			using input_type = bool;

			template<typename InputType>
			using output_type = bool;
		};
	}
}

#endif //VPROMOTE_H
