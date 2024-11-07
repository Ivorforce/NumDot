#ifndef VCOMPUTE_INPLACE_H
#define VCOMPUTE_INPLACE_H

#include <cmath>                       // for double_t
#include <cstdint>                     // for int64_t
#include <type_traits>                  // for decay_t, conditional_t, disju...
#include <utility>                      // for forward
#include <variant>                      // for visit, variant
#include <vector>                       // for vector
#include "varray.hpp"                     // for VArrayTarget, VScalar, VData
#include "vassign.hpp"                    // for assign_nonoverlapping, broadc...
#include "vpromote.hpp"                   // for promote_value_type_if_needed
#include "vconfig.hpp"
#include "xtensor/xarray.hpp"           // for xarray_container
#include "xtensor/xoperation.hpp"       // for xfunction_type_t
#include "xtensor/xstorage.hpp"         // for uvector
#include "xtensor/xtensor_forward.hpp"  // for xarray

namespace va {
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

    template<typename FX>
    struct XFunction {
        // This is analogous to xt::add etc., with the main difference that in our setup it's easier to use this function with the
        //  appropriate xt::detail:: operation.
        template<typename... Args>
        inline auto operator()(Args&&... args) const -> xt::detail::xfunction_type_t<FX, Args...> {
            return xt::detail::make_xfunction<FX>(std::forward<Args>(args)...);
        }
    };

    template <Feature feature, class Visitor, class... Vs>
    constexpr auto visit_if_enabled(Visitor&& visitor, Vs&&... vs) -> decltype(std::visit(visitor, vs...)) {
        if constexpr (va::is_feature_enabled(feature)) {
            return std::visit(std::forward<Visitor>(visitor), std::forward<Vs>(vs)...);
        }
        else {
            // TODO add what feature is missing
            throw std::runtime_error(std::string("Function execution failed: Missing feature in NumDot build - ") + feature_name(feature));
        }
    }

    template<typename OutputType, typename Result>
    std::shared_ptr<VArray> create_varray(VStoreAllocator& allocator, const Result& result) {
        using RNatural = typename std::decay_t<decltype(result)>::value_type;
        using OStorable = compatible_type_or_64_bit_t<OutputType, VScalar>;

        static_assert(std::is_convertible_v<RNatural, OStorable>, "Cannot store the function result.");

        const auto dimension = result.dimension();

        shape_type shape(dimension);
        std::copy_n(result.shape().begin(), dimension, shape.begin());

        // Create new array, assign to our target pointer.
        // OutputType may be different from R, if we want different behavior than xtensor for computation.
        std::shared_ptr<VStore> result_store = allocator.allocate(va::dtype_of_type<OStorable>(), result.size());
        auto data = make_compute<OStorable*>(
            static_cast<OStorable*>(result_store->data()),
            shape,
            strides_type{}, // unused
            dimension <= 1 ? xt::layout_type::any : xt::layout_type::row_major
        );
        va::broadcasting_assign(data, result);

        return std::make_shared<VArray>(
            VArray {
                std::move(result_store),
                std::move(data),
                0
            }
        );
    }

    template<typename OutputType, typename Result>
    void assign_to_target(VArrayTarget target, VStoreAllocator& allocator, const Result& result) {
        // Some functions (or compilers) may offer 128 bit types as results from functions.
        // We may not be able to store them. This is not the best way to go about it
        // (as incompatible types COULD be less than the supported ones, prompting us to upcast), but it's good enough for now.
        using RNatural = typename std::decay_t<decltype(result)>::value_type;
        using RStorable = compatible_type_or_64_bit_t<RNatural, VScalar>;

        std::visit(
            [&result, &allocator](auto& target) {
                using PtrType = std::decay_t<decltype(target)>;

#ifndef NUMDOT_COPY_FOR_ALL_INPLACE_OPERATIONS
                if constexpr (std::is_same_v<PtrType, VData*>) {
                    // Assign to compute case, broadcasting and casting if necessary.
                    if (std::visit(
                        [&result](auto& ctarget) {
                            using T = typename std::decay_t<decltype(ctarget)>::value_type;

#ifndef NUMDOT_OPTIMIZE_ALL_INPLACE_OPERATIONS
                            // If the target type is the same as the natural operation type,
                            // we can just assign. It's just one operation, and it's the most important one!
                            // Otherwise, we may want to avoid it because it generates a lot of code (nxm).
                            if constexpr (!std::is_same_v<T, RNatural>) {
                                // We need to cast, just give up here.
                                return false;
                            }
                            else
#endif
                            {
                                // TODO Could use assign_xexpression if there is no aliasing, aka overlap of target and inputs.
                                va::broadcasting_assign(ctarget, result);
                                return true;
                            }
                        }, *target
                    )) {
                        // Ran accelerated assign, we don't need to do the regular one.
                        return;
                    }
                    else
#endif
                    {
                        if constexpr (!std::is_convertible_v<RNatural, RStorable>) {
                            throw std::runtime_error("Cannot store the function result in an array of this dtype.");
                        }
                        else {
                            // Make a copy, similar as in promote_compute_case_if_needed.
                            // This is the same operation on all branches so we don't produce much code.
                            const auto temporary = create_varray<RStorable>(allocator, result);
                            // Now, a normal type conversion.
                            va::assign(*target, temporary->data);
                        }
                    }
                }
                else {
                    *target = create_varray<OutputType>(allocator, result);
                }
            }, target
        );
    }

    template<typename PromotionRule, typename Visitor>
    struct VArrayFunctionInplace {
        const Visitor visitor;
        const VArrayTarget target;
        VStoreAllocator& allocator;

        explicit VArrayFunctionInplace(const Visitor visitor, const VArrayTarget target, VStoreAllocator& allocator)
            : visitor(std::move(visitor)), target(target), allocator(allocator) {}

        template<typename... Args>
        void operator()(const Args&... args) {
            using InputType = typename PromotionRule::template input_type<promote::value_type_v<std::decay_t<Args>>...>;

            if constexpr (std::is_same_v<InputType, void>) {
                throw std::runtime_error("Unsupported type for operation.");
            }
            else if constexpr (!std::disjunction_v<std::is_convertible<promote::value_type_v<std::decay_t<Args>>, InputType>...>) {
                throw std::runtime_error("Cannot promote in this way.");
            }
            else {
                // Result of visitor invocation
                const auto result = visitor(promote::promote_value_type_if_needed<InputType>(args)...);

                using NaturalOutputType = typename std::decay_t<decltype(result)>::value_type;
                using OutputType = typename PromotionRule::template output_type<InputType, NaturalOutputType>;

                assign_to_target<OutputType>(target, allocator, result);
            }
        }
    };

    template<Feature feature, typename PromotionRule, typename FX, typename... Args>
    static inline void xoperation_inplace(FX&& fx, VStoreAllocator& allocator, VArrayTarget target, const Args&... args) {
        visit_if_enabled<feature>(
            VArrayFunctionInplace<PromotionRule, FX> { std::forward<FX>(fx), target, allocator },
            args...
        );
    }

    template<typename PromotionRule, typename ReturnType, typename Visitor>
    struct VArrayReduction {
        const Visitor visitor;

        explicit VArrayReduction(const Visitor visitor)
            : visitor(std::move(visitor)) {}

        template<typename... Args>
        ReturnType operator()(const Args&... args) const {
            using InputType = typename PromotionRule::template input_type<typename std::decay_t<Args>::value_type...>;

            if constexpr (std::is_same_v<InputType, void>) {
                throw std::runtime_error("Unsupported type for operation.");
            }
            else if constexpr (!std::disjunction_v<std::is_convertible<promote::value_type_v<std::decay_t<Args>>, InputType>...>) {
                throw std::runtime_error("Cannot promote in this way.");
            }
            else {
                using NaturalOutputType = decltype(visitor(promote::promote_value_type_if_needed<InputType>(args)...));
                using OutputType = typename PromotionRule::template output_type<InputType, NaturalOutputType>;

                // Result of visitor invocation
                // TODO Some xt functions support passing the output type. That would be FAR better than casting it afterwards as here.
                const auto result = OutputType(visitor(promote::promote_value_type_if_needed<InputType>(args)...));
                return static_cast<ReturnType>(result);
            }
        }
    };

    template<Feature feature, typename PromotionRule, typename ReturnType, typename FX, typename... Args>
    static ReturnType vreduce(FX&& fx, const Args&... args) {
        return visit_if_enabled<feature>(
            VArrayReduction<PromotionRule, ReturnType, FX> { std::forward<FX>(fx) },
            args...
        );
    }
}

#endif //VCOMPUTE_INPLACE_H
