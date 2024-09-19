#ifndef VCOMPUTE_INPLACE_H
#define VCOMPUTE_INPLACE_H

#include "varray.h"
#include "vpromote.h"

namespace va {
    template<typename FX>
    struct XFunction {
        // This is analogous to xt::add etc., with the main difference that in our setup it's easier to use this function with the
        //  appropriate xt::detail:: operation.
        template<typename... Args>
        inline auto operator()(Args &&... args) const -> xt::detail::xfunction_type_t<FX, Args...> {
            return xt::detail::make_xfunction<FX>(std::forward<Args>(args)...);
        }
    };

    template<typename OutputType, typename Result>
    void assign_to_target(VArrayTarget target, Result&& result) {
        using R = typename std::decay_t<decltype(result)>::value_type;

        std::visit([&result](auto&& target) {
            using PtrType = std::decay_t<decltype(target)>;

            if constexpr (std::is_same_v<PtrType, ComputeVariant *>) {
                // Assign to compute case, broadcasting and casting if necessary.
                std::visit([&result](auto &&ctarget) {
                    using T = typename std::decay_t<decltype(ctarget)>::value_type;
#ifdef NUMDOT_ASSIGN_INPLACE_DIRECTLY_INSTEAD_OF_COPYING_FIRST

                    // About 30% of binary size is the first case, because assignment to a compute case can be difficult.
                    // Another 20% is the two cases combined, which are inlined because they both assign directly to a new xarray.
                    if constexpr (std::is_same_v<T, R>) {
                        // TODO Could use assign_xexpression if there is no aliasing, aka overlap of target and inputs.
                        ctarget.computed_assign(result);
                    } else
#endif
                    {
                        // Make a copy, similar as in promote_compute_case_if_needed.
                        // After copying we can be sure no aliasing is taking place, so we can assign with assign_xexpression.
                        ctarget.assign_xexpression(xt::xarray<R>(result));
                    }
                }, *target);
            } else {
                // Create new array, assign to our target pointer.
                // OutputType may be different from R, if we want different behavior than xtensor for computation.
                *target = from_store(std::make_shared<xt::xarray<OutputType>>(result));
            }
        }, target);
    }

    template<typename PromotionRule, typename Visitor>
    struct VArrayFunctionInplace {
        const Visitor visitor;
        const VArrayTarget target;

        explicit VArrayFunctionInplace(const Visitor visitor, const VArrayTarget target)
            : visitor(std::move(visitor)), target(target) {
        }

        template<typename... Args>
        void operator()(const compute_case<Args>&... args) const {
            using InputType = typename PromotionRule::template input_type<Args...>;
            using OutputType = typename PromotionRule::template output_type<InputType>;

            // Result of visitor invocation
            const auto result = visitor(promote::promote_compute_case_if_needed<InputType>(args)...);

            assign_to_target<OutputType>(target, result);
        }
    };

    template<typename PromotionRule, typename FX, typename... Args>
    static inline void xoperation_inplace(FX &&fx, VArrayTarget target, const Args&... args) {
        std::visit(
            VArrayFunctionInplace<PromotionRule, FX>{std::forward<FX>(fx), target },
            args...
        );
    }

    template<typename PromotionRule, typename FX>
    static VArrayFunctionInplace<PromotionRule, FX> make_varrayfunction_inplace(FX fx, VArrayTarget target) {
        return VArrayFunctionInplace<PromotionRule, FX>{fx, target};
    }

    template<typename PromotionRule, typename FX, typename Axes, typename... Args>
    static inline void xreduction_inplace(FX &&fx, Axes &&axes, VArrayTarget target, Args&&... args) {
        std::visit([fx = std::forward<FX>(fx), target](auto& axes, auto&&... stores) {
            using AxesType = std::decay_t<decltype(axes)>;

            if constexpr (std::is_same_v<AxesType, std::nullptr_t>) {
                make_varrayfunction_inplace<PromotionRule>([fx](auto&&... inner_args) {
                    return fx(std::forward<decltype(inner_args)>(inner_args)...);
                }, target)(std::forward<decltype(stores)>(stores)...);
            } else {
                make_varrayfunction_inplace<PromotionRule>([fx, &axes](auto&&... inner_args) {
                    return fx(axes, std::forward<decltype(inner_args)>(inner_args)...);
                }, target)(std::forward<decltype(stores)>(stores)...);
            }
        }, std::forward<Axes>(axes), std::forward<Args>(args)...);
    }
}

#endif //VCOMPUTE_INPLACE_H
