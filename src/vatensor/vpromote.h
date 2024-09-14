#ifndef VPROMOTE_H
#define VPROMOTE_H

namespace va {
    namespace promote {
        template<typename Arg>
        using int64_if_bool_else_id = typename std::conditional<
            std::is_same_v<Arg, bool>,
            int64_t,
            Arg
        >::type;

        // TODO We may want to support mixed-type input ops for some functions, to avoid explicitly promoting types.
        //  I think it may be faster to not cast beforehand, but it's possible it does it later down the line anyway.
        //  That should be tested.
        // Also, mixed-type input ops can really increase binary size, so it should be used with care if at all.

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

        struct num_at_least_int32 {
            template<typename Arg>
            using input_type = typename std::conditional<
                (std::numeric_limits<int64_if_bool_else_id<Arg>>::digits >= std::numeric_limits<int32_t>::digits),
                int64_if_bool_else_id<Arg>,
                typename std::conditional<
                    std::is_signed<int64_if_bool_else_id<Arg>>::value,
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
                std::is_floating_point_v<std::common_type_t<int64_if_bool_else_id<Args>...> >,
                std::common_type_t<int64_if_bool_else_id<Args>...>,
                Default
            >;

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