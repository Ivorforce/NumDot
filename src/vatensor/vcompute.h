#ifndef XTVA_H
#define XTVA_H

#include "varray.h"

namespace va {
    template<typename PromotionRule, typename Visitor>
    struct VArrayFunction {
        Visitor visitor;

        explicit VArrayFunction(const Visitor &visitor)
            : visitor(visitor) {
        }

        template <typename Arg>
        auto promote_if_needed(const compute_case<Arg>& arg) const {
            using ArgPostPromote = typename PromotionRule::template promote_input<Arg>;

            if constexpr (std::is_same_v<ArgPostPromote, Arg>) {
                return arg;
            }
            else {
                return xt::cast<ArgPostPromote>(arg);
            }
        }

#ifndef NUMDOT_ALLOW_MIXED_TYPE_OPS
        // This version exists to reduce the number of functions generated, from O(op n n) to O(o n).
        // Essentially, we make sure that all calls to the function are performed with all types being the same.
        // Every type that doesn't fit will be promoted before the function call.

        // You may think this unnecessary, but it actually reduces the binary size by a factor of 2-3.
        // All 'good' cases, where a promotion is not necessary, retain the same speed.

        // TODO Regain perfect forwarding. Somehow, last time I tried it didn't work properly.
        template<typename A, typename B>
        VArray operator()(const compute_case<A>& a, const compute_case<B>& b) const {
            using ResultType = typename PromotionRule::template result<
                typename PromotionRule::template promote_input<A>,
                typename PromotionRule::template promote_input<B>
            >;

            // Note: If we made a copy instead of just casting, we could save even more space and compile time.
            //  But it also comes at a larger cost. Just casting as type should be the best of both worlds.
            if constexpr (std::is_same_v<A, B>) {
                // The types are the same, we can just call. If they're wrong, xtensor will promote them for us with optimal performance.
                auto result = visitor(promote_if_needed(a), promote_if_needed(b));
                return from_store(std::make_shared<xt::xarray<ResultType>>(result));
            } else if constexpr (std::is_same_v<A, ResultType>) {
                // a is good, promote b.
                auto result = visitor(promote_if_needed(a), xt::xarray<ResultType>(xt::cast<ResultType>(b)));
                return from_store(std::make_shared<xt::xarray<ResultType>>(result));
            } else if constexpr (std::is_same_v<B, ResultType>) {
                // b is good, promote a.
                auto result = visitor(xt::cast<ResultType>(a), promote_if_needed(b));
                return from_store(std::make_shared<xt::xarray<ResultType>>(result));
            } else {
                // Both are bad, promote both.
                auto result = visitor(xt::cast<ResultType>(a), xt::cast<ResultType>(b));
                return from_store(std::make_shared<xt::xarray<ResultType>>(result));
            }
        }
#endif

        template<typename... Args>
        VArray operator()(const compute_case<Args>&... args) const {
            using ResultType = typename PromotionRule::template result<
                typename PromotionRule::template promote_input<Args>...
            >;

            // This doesn't do anything yet, it just constructs a value for operation.
            // It will be executed when we use it on the xarray constructor!
            auto result = visitor(promote_if_needed(args)...);

            // Note: Need to do this in one line. If the operator is called after the make_shared,
            //  any situations where broadcast errors would be thrown will instead crash the program.
            return from_store(std::make_shared<xt::xarray<ResultType>>(result));
        }
    };

    template<typename FX>
    struct XFunction {
        // This is analogous to xt::add etc., with the main difference that in our setup it's easier to use this function with the
        //  appropriate xt::detail:: operation.
        template<typename... Args>
        inline auto operator()(Args &&... args) const -> xt::detail::xfunction_type_t<FX, Args...> {
            return xt::detail::make_xfunction<FX>(std::forward<Args>(args)...);
        }
    };

    template<typename PromotionRule, typename FX, typename... Args>
    static inline VArray xoperation(FX &&fx, const Args&... args) {
        return std::visit(VArrayFunction<PromotionRule, FX>{std::forward<FX>(fx)}, args...);
    }

    template<typename PromotionRule, typename FX>
    static VArrayFunction<PromotionRule, FX> make_varrayfunction(FX fx) {
        return VArrayFunction<PromotionRule, FX>{fx};
    }

    template<typename PromotionRule, typename FX, typename Axes, typename... Args>
    static inline VArray xreduction(FX &&fx, Axes &&axes, Args &&... args) {
        return std::visit([fx = std::forward<FX>(fx)](auto axes, auto&&... stores) {
            using AxesType = std::decay_t<decltype(axes)>;

            // TODO I think we can perfect forward better, but for now I can't get it to work.
            if constexpr (std::is_same_v<AxesType, std::nullptr_t>) {
                return make_varrayfunction<PromotionRule>([fx](auto... inner_args) {
                    return fx(inner_args...);
                })(stores...);
            } else {
                return make_varrayfunction<PromotionRule>([fx, axes](auto... inner_args) {
                    return fx(axes, inner_args...);
                })(stores...);
            }
        }, std::forward<Axes>(axes), std::forward<Args>(args)...);
    }
}

#endif //XTVA_H
