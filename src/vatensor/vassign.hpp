#ifndef NUMDOT_VASSIGN_H
#define NUMDOT_VASSIGN_H

#include "varray.hpp"               // for VData, VScalar, ArrayVariant, VArr...
#include "xtensor/xassign.hpp"    // for assert_compatible_shape, assign_data
#include "xtensor/xsemantic.hpp"  // for get_rhs_triviality

namespace xt {
	template<class D>
	class xexpression;
}

namespace va {
	// computed_assign on containers doesn't assign data, it tries to assign to the whole container.
	// This is basically view_semantic's computed_assign.
	template<typename T, typename E>
	inline void broadcasting_assign(xt::xexpression<T>& t, const xt::xexpression<E>& e) {
		xt::assert_compatible_shape(t, e);
		xt::assign_data(t, e, xt::detail::get_rhs_triviality(e.derived_cast()));
	}

	// Some implicit casts are not possible, see https://github.com/xtensor-stack/xtensor/issues/2815.
	template<typename R, typename E>
	inline void broadcasting_assign_typesafe(R& t, const E& e) {
		using RT = typename std::decay_t<decltype(t)>::value_type;
		// using ET = typename std::decay_t<decltype(e)>::value_type;

		if constexpr (std::is_same_v<RT, bool>) va::broadcasting_assign(t, xt::cast<bool>(e));
		else va::broadcasting_assign(t, e);
	}

	void set_single_value(VData& array, axes_type& index, const VScalar& value);
	VScalar get_single_value(const VData& array, axes_type& index);

	void assign(VStoreAllocator& allocator, const VArrayTarget& target, const VData& value);
	void assign_cast(VStoreAllocator& allocator, const VArrayTarget& target, const VData& value, DType dtype);
	void assign(const VArrayTarget& target, VScalar value);

	std::shared_ptr<VArray> get_at_mask(VStoreAllocator& allocator, const VData& varray, const VData& mask);
	void set_at_mask(VData& varray, VData& mask, VData& value);
	void set_at_mask(VData& varray, VData& mask, VScalar value);

	std::shared_ptr<VArray> get_at_indices(VStoreAllocator& allocator, const VData& varray, const VData& indices);
	void set_at_indices(VData& varray, VData& indices, VData& value);
}

#endif //NUMDOT_VASSIGN_H
