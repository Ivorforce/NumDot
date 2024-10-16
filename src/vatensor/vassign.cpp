#include "vassign.hpp"

#include <type_traits>                                  // for decay_t
#include <variant>                                      // for visit
#include "varray.hpp"                            // for VWrite, VScalar
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xindex_view.hpp>

#include "allocate.hpp"

using namespace va;

static void mod_index(axes_type& index, const shape_type& shape) {
	// xtensor actually checks later, too, but it just pads with 0 rather than throwing.
	if (index.size() != shape.size()) throw std::runtime_error("invalid dimension for index");

	for (int i = 0; i < index.size(); ++i) {
		if (index[i] < 0) index[i] += shape[i];
		if (index[i] < 0 || index[i] >= shape[i]) throw std::runtime_error("index out of bounds");
	}
}

void va::set_single_value(VWrite& array, axes_type& index, VScalar value) {
	std::visit(
		[&index](auto& carray, auto value) -> void {
			mod_index(index, carray.shape());
			carray[index] = value;
		}, array, value
	);
}

VScalar va::get_single_value(VRead& array, axes_type& index) {
	return std::visit(
		[&index](auto& carray) -> VScalar {
			mod_index(index, carray.shape());
			return carray[index];
		}, array
	);
}

void va::assign(VWrite& array, const VRead& value) {
	if (va::dimension(value) == 0) {
		// Optimization for
		va::assign(array, va::to_single_value(value));
		return;
	}

	std::visit(
		[](auto& carray, const auto& cvalue) {
			using VWrite = typename std::decay_t<decltype(carray)>::value_type;
			using VRead = typename std::decay_t<decltype(carray)>::value_type;

#ifdef XTENSOR_USE_XSIMD
			// For some reason, bool - to - bool assignments are broken in xsimd
			// TODO Should make this reproducible, I haven't managed so far.
			// See https://github.com/Ivorforce/NumDot/issues/123
			if constexpr (std::is_same_v<VWrite, bool> && std::is_same_v<VRead, bool>) {
				broadcasting_assign(carray, xt::cast<uint8_t>(cvalue));
			}
			else
#endif
			{
				broadcasting_assign(carray, cvalue);
			}
		}, array, value
	);
}

void va::assign_nonoverlapping(VWrite& array, const ArrayVariant& value) {
	std::visit(
		[](auto& carray, const auto& cvalue) {
			using VWrite = typename std::decay_t<decltype(carray)>::value_type;
			using VRead = typename std::decay_t<decltype(carray)>::value_type;

#ifdef XTENSOR_USE_XSIMD
			// See above
			if constexpr (std::is_same_v<VWrite, bool> && std::is_same_v<VRead, bool>) {
				broadcasting_assign(carray, xt::cast<uint8_t>(cvalue));
			}
			else
#endif
			{
				broadcasting_assign(carray, cvalue);
			}
		}, array, value
	);
}

void va::assign(VWrite& array, VScalar value) {
	std::visit(
		[](auto& carray, const auto cvalue) {
			using T = std::decay_t<decltype(carray)>;
			using V = typename T::value_type;

			// TODO The .fill makes this check statically only, so is_contiguous() isn't called.
			// See https://github.com/xtensor-stack/xtensor/pull/2809
			// carray.fill(static_cast<V>(cvalue));
			if (T::contiguous_layout || carray.is_contiguous())
			{
				std::fill(carray.linear_begin(), carray.linear_end(), static_cast<V>(cvalue));
			}
			else
			{
				std::fill(carray.begin(), carray.end(), static_cast<V>(cvalue));
			}
		}, array, value
	);
}

void va::assign(VArrayTarget target, const VRead& value) {
	std::visit(
		[&value](auto target) {
			if constexpr (std::is_same_v<decltype(target), VWrite*>) {
				va::assign(*target, value);
			}
			else {
				*target = va::copy(value);
			}
		}, target
	);
}

void va::assign(VArrayTarget target, VScalar value) {
	std::visit(
		[value](auto target) {
			if constexpr (std::is_same_v<decltype(target), VWrite*>) {
				va::assign(*target, value);
			}
			else {
				*target = from_scalar_variant(value);
			}
		}, target
	);
}

std::shared_ptr<VArray> va::get_at_mask(const VRead& varray, const VRead& mask) {
#ifdef NUMDOT_DISABLE_INDEX_MASKS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_INDEX_MASKS to enable it.");
#else
	return std::visit(
		[](const auto& array, const auto& mask) -> std::shared_ptr<VArray> {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else {
				// Masked views don't offer this functionality automatically.
				const size_type array_size = xt::sum(mask)();
				auto result = make_store<VTArray>(xt::empty<VTArray>({ array_size }));
				const auto masked_view = xt::masked_view(array, mask);

				auto iter_result = result->begin();
				for (auto masked_value : masked_view) {
					if (masked_value.visible()) {
						*iter_result = masked_value.value();
						++iter_result;
					}
				}

				return from_store(result);
			}
		}, varray, mask
	);
#endif
}

void va::set_at_mask(VWrite& varray, VRead& mask, VRead& value) {
#ifdef NUMDOT_DISABLE_INDEX_MASKS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_INDEX_MASKS to enable it.");
#else
	// This case is not handled again later, so we actually need this 'performance' check.
    if (va::dimension(value) == 0) {
	    return set_at_mask(varray, mask, va::to_single_value(value));
    }

	return std::visit(
		// Mask can't be const because of masked_view iterator.
		[](auto& array, auto& mask, const auto& value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else {
				if (array.shape() != mask.shape()) {
					throw std::runtime_error("mask must be same shape as array");
				}

				auto masked_view = xt::masked_view(array, mask);

				// Masked views don't offer array fill functionality automatically.
				const size_type array_size = xt::sum(mask)();
				if (value.shape() != shape_type { array_size })
					throw std::runtime_error("mask must be single value or match the mask sum");

				const auto stride = value.strides()[0];
				auto iter_value = value.begin();
				for (auto masked_value : masked_view) {
					if (masked_value.visible()) {
						masked_value.value() = static_cast<VTArray>(*iter_value);
						iter_value += stride;
					}
				}
			}
		}, varray, mask, value
	);
#endif
}

void va::set_at_mask(VWrite& varray, VRead& mask, VScalar value) {
	return std::visit(
		// Mask can't be const because of masked_view iterator.
		[](auto& array, auto& mask, const auto& value) {
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else {
				if (array.shape() != mask.shape()) {
					throw std::runtime_error("mask must be same shape as array");
				}

				auto masked_view = xt::masked_view(array, mask);
				masked_view = value;
			}
		}, varray, mask, value
	);
}

template<typename A>
xt::svector<xt::svector<size_type>> array_to_indices(const A& indices) {
	const auto num_indices = indices.shape()[0];
	const auto num_dimensions = indices.shape()[1];
	const auto& strides = indices.strides();

	// TODO This should be possible without allocating separately for each xidx, no?
	xt::svector<xt::svector<size_type>> xindices(num_indices);
	for (int i = 0; i < xindices.size(); ++i) {
		xt::svector<size_type> xidx(num_dimensions);
		std::copy(indices.begin() + i * strides[0], indices.begin() + (i + 1) * strides[0], xidx.begin());
		xindices[i] = xidx;
	}
	return xindices;
}

std::shared_ptr<VArray> va::get_at_indices(const VRead& varray, const VRead& indices) {
#ifdef NUMDOT_DISABLE_INDEX_LISTS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_INDEX_LISTS to enable it.");
#else
	return std::visit(
		[](const auto& array, const auto& indices) -> std::shared_ptr<VArray> {
			using VTMask = typename std::decay_t<decltype(indices)>::value_type;

			if constexpr (!std::is_integral_v<VTMask> || std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be integer dtype");
			}
			else {
				if (indices.dimension() == 1) {
					if (array.dimension() != 1) throw std::runtime_error("cannot use 1D index list for nd tensor");

					return from_store(make_store(xt::index_view(array, indices)));
				}
				if (indices.dimension() != 2) throw std::runtime_error("index list must be 1d or 2d");
				if (indices.shape()[1] != array.dimension()) throw std::runtime_error("index list dimension 2 must match array dimension");

				// Index views need to be vectors of xindex.
				xt::svector<xt::svector<size_type>> xindices = array_to_indices(indices);
				return from_store(make_store(xt::index_view(array, xindices)));
			}
		}, varray, indices
	);
#endif
}

void va::set_at_indices(VWrite& varray, VRead& indices, VRead& value) {
#ifdef NUMDOT_DISABLE_INDEX_LISTS
	throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_INDEX_LISTS to enable it.");
#else
	std::visit(
		[](auto& array, const auto& indices, const auto& value) {
			using VTMask = typename std::decay_t<decltype(indices)>::value_type;

			if constexpr (!std::is_integral_v<VTMask> || std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be integer dtype");
			}
			else {
				if (indices.dimension() == 1) {
					if (array.dimension() != 1) throw std::runtime_error("cannot use 1D index list for nd tensor");

					auto index_view = xt::index_view(array, indices);
					index_view = value;
					return;
				}
				if (indices.dimension() != 2) throw std::runtime_error("index list must be 1d or 2d");
				if (indices.shape()[1] != array.dimension()) throw std::runtime_error("index list dimension 2 must match array dimension");

				// Index views need to be vectors of xindex.
				xt::svector<xt::svector<size_type>> xindices = array_to_indices(indices);
				auto index_view = xt::index_view(array, xindices);
				index_view = value;
			}
		}, varray, indices, value
	);
#endif
}
