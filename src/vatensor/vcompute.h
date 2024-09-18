#ifndef XTVA_H
#define XTVA_H

#include "varray.h"

namespace va {
    template <typename NeededType, typename Type>
    auto promote_compute_case_if_needed(const compute_case<Type>& arg) {
        if constexpr (std::is_same_v<Type, NeededType>) {
            // Most common situation: the argument we need is the same as the argument that's given.
            return arg;
        }
        else {
            // Casting can considerably increase performance (from a small test, it was 25%).
            // However, this is only relevant for operations that even need casting.
            // The cost for casting instead of copying is a much larger binary size (100% increase).
            // Most people will probably prefer the small binary, and accept less optimized wrong dtype operations.
#ifdef NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS
            return xt::cast<NeededType>(arg);
#else
            return xt::xarray<NeededType>(arg);
#endif
        }
    }

    template<typename PromotionRule, typename Visitor>
    struct VArrayFunction {
        Visitor visitor;

        explicit VArrayFunction(const Visitor &visitor)
            : visitor(visitor) {
        }

        // TODO Regain perfect forwarding. Somehow, last time I tried it didn't work properly.
        template<typename... Args>
        VArray operator()(const compute_case<Args>&... args) const {
            using InputType = typename PromotionRule::template input_type<Args...>;
            using OutputType = typename PromotionRule::template output_type<InputType>;

            // This doesn't do anything yet, it just constructs a value for operation.
            // It will be executed when we use it on the xarray constructor!
            auto result = visitor(promote_compute_case_if_needed<InputType>(args)...);

            // Note: Need to do this in one line. If the operator is called after the make_shared,
            //  any situations where broadcast errors would be thrown will instead crash the program.
            return from_store(std::make_shared<xt::xarray<OutputType>>(result));
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
