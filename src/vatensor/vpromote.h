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

        /**
         * 	what results from the native C++ operation op(A(), B())
         */
        template<typename FN>
        struct num_function_result {
            template<typename Arg>
            using promote_input = int64_if_bool_else_id<Arg>;

            template<typename... Args>
            using result = decltype(std::declval<FN>()(std::forward<Args>(std::declval<Args>())...));
        };

        struct num_common_type {
            template<typename Arg>
            using promote_input = int64_if_bool_else_id<Arg>;

            template<typename... Args>
            using result = std::common_type_t<Args...>;
        };

        struct num_at_least_int32 {
            template<typename Arg>
            using promote_input = int64_if_bool_else_id<Arg>;

            template<typename T>
            using result = typename std::conditional<
                (std::numeric_limits<T>::digits >= std::numeric_limits<int32_t>::digits),
                T,
                typename std::conditional<
                    std::is_signed<T>::value,
                    int32_t,
                    uint32_t
                >::type
            >::type;
        };

        template<typename Default>
        struct num_matching_float_or_default {
            template<typename Arg>
            using promote_input = int64_if_bool_else_id<Arg>;

            template<typename... Args>
            using result = std::conditional_t<
                std::is_floating_point_v<std::common_type_t<Args...> >,
                std::common_type_t<Args...>,
                Default
            >;
        };

        struct identity_in_bool_out {
            template<typename Arg>
            using promote_input = Arg;

            template<typename... Args>
            using result = bool;
        };

        struct bool_in_bool_out {
            template<typename Arg>
            using promote_input = bool;

            template<typename... Args>
            using result = bool;
        };
    }
}

#endif //VPROMOTE_H
