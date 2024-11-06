#include "vassign.hpp"

#include <type_traits>                                  // for decay_t
#include <variant>                                      // for visit
#include "varray.hpp"                            // for VData, VScalar
#include "create.hpp"
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xindex_view.hpp>
#include "vcompute.hpp"
#include "vpromote.hpp"
#include "xscalar_store.hpp"

using namespace va;

static void mod_index(axes_type& index, const shape_type& shape) {
	// xtensor actually checks later, too, but it just pads with 0 rather than throwing.
	if (index.size() != shape.size()) throw std::runtime_error("invalid dimension for index");

	for (int i = 0; i < index.size(); ++i) {
		if (index[i] < 0) index[i] += shape[i];
		if (index[i] < 0 || index[i] >= shape[i]) throw std::runtime_error("index out of bounds");
	}
}

void va::set_single_value(VData& array, axes_type& index, VScalar value) {
	std::visit(
		[&index](auto& carray, auto value) -> void {
			if constexpr (!std::is_convertible_v<decltype(value), typename std::decay_t<decltype(carray)>::value_type>) {
				throw std::runtime_error("Cannot promote in this way.");
			}
			else {
				mod_index(index, carray.shape());
				carray[index] = value;
			}
		}, array, value
	);
}

VScalar va::get_single_value(const VData& array, axes_type& index) {
	return std::visit(
		[&index](auto& carray) -> VScalar {
			mod_index(index, carray.shape());
			return carray[index];
		}, array
	);
}

void va::assign(VData& array, const VData& value) {
	if (va::dimension(value) == 0) {
		// Optimization for 0D tensors
		va::assign(array, va::to_single_value(value));
		return;
	}

	std::visit(
		[](auto& carray, const auto& cvalue) {
			using VWrite = typename std::decay_t<decltype(carray)>::value_type;
			using VRead = typename std::decay_t<decltype(cvalue)>::value_type;

			if constexpr (!std::is_convertible_v<VRead, VWrite>) {
				throw std::runtime_error("Cannot promote in this way.");
			}
#ifdef XTENSOR_USE_XSIMD
			// For some reason, bool - to - bool assignments are broken in xsimd
			// TODO Should make this reproducible, I haven't managed so far.
			// See https://github.com/Ivorforce/NumDot/issues/123
			else if constexpr (std::is_same_v<VWrite, bool> && std::is_same_v<VRead, bool>) {
				broadcasting_assign(carray, xt::cast<uint8_t>(cvalue));
			}
			else if constexpr (xtl::is_complex<VWrite>::value) {
				// xsimd also has no auto conversion into complex types
				broadcasting_assign(carray, xt::cast<VWrite>(cvalue));
			}
#endif
			else
			{
				broadcasting_assign(carray, cvalue);
			}
		}, array, value
	);
}

void va::assign(VData& array, VScalar value) {
	std::visit(
		[](auto& carray, const auto cvalue) {
			using T = std::decay_t<decltype(carray)>;
			using V = typename T::value_type;

			if constexpr (!std::is_convertible_v<decltype(cvalue), V>) {
				throw std::runtime_error("Cannot promote in this way.");
			}
			else {
				const auto value = static_cast<V>(cvalue);

				// TODO The .fill makes this check statically only, so is_contiguous() isn't called.
				// See https://github.com/xtensor-stack/xtensor/pull/2809
				// carray.fill(static_cast<V>(cvalue));
				if (T::contiguous_layout || carray.is_contiguous())
				{
					std::fill(carray.linear_begin(), carray.linear_end(), value);
				}
				else
				{
					std::fill(carray.begin(), carray.end(), value);
				}
			}
		}, array, value
	);
}

void va::assign(VStoreAllocator& allocator, VArrayTarget target, const VData& value) {
	std::visit(
		[&value, &allocator](auto target) {
			if constexpr (std::is_same_v<decltype(target), VData*>) {
				va::assign(*target, value);
			}
			else {
				*target = va::copy(allocator, value);
			}
		}, target
	);
}

void va::assign_cast(VStoreAllocator& allocator, VArrayTarget target, const VData& value, DType dtype) {
	if (va::dtype(value) == dtype) {
		// No cast necessary, just assign.
		va::assign(allocator, target, value);
		return;
	}

	std::visit(
		[&allocator, &value, dtype](auto target) {
			if constexpr (std::is_same_v<decltype(target), VData*>) {
				// Making a copy is slow, but it's also a bit unusual to in-place
				// assign to a type that has another type.
				const auto casted_copy = copy_as_dtype(allocator, value, dtype);
				va::assign(*target, casted_copy->data);
			}
			else {
				*target = copy_as_dtype(allocator, value, dtype);
			}
		}, target
	);
}

void va::assign(VArrayTarget target, VScalar value) {
	std::visit(
		[value](auto target) {
			if constexpr (std::is_same_v<decltype(target), VData*>) {
				va::assign(*target, value);
			}
			else {
				*target = va::store::from_scalar_variant(value);
			}
		}, target
	);
}

std::shared_ptr<VArray> va::get_at_mask(VStoreAllocator& allocator, const VData& data, const VData& mask) {
	return visit_if_enabled<Feature::index_masks>(
		[&allocator, &data](const auto& mask) -> std::shared_ptr<VArray> {
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else {
				const size_type array_size = xt::sum(mask)();

				return std::visit([&mask, &allocator, array_size](const auto& array) -> std::shared_ptr<VArray> {
					using VTArray = typename std::decay_t<decltype(array)>::value_type;

					auto result_varray = va::empty(allocator, variant_to_dtype(VTArray{}), shape_type { array_size });
					auto result_compute = std::get<compute_case<VTArray*>>(result_varray->data);

					// Masked views don't offer this functionality automatically.
					const auto masked_view = xt::masked_view(array, mask);

					auto iter_result = result_compute.begin();
					for (auto masked_value : masked_view) {
						if (masked_value.visible()) {
							*iter_result = masked_value.value();
							++iter_result;
						}
					}

					return result_varray;
				}, data);
			}
		}, mask
	);
}

void va::set_at_mask(VData& varray, VData& mask, VData& value) {
	// This case is not handled again later, so we actually need this 'performance' check.
    if (va::dimension(value) == 0) {
	    return set_at_mask(varray, mask, va::to_single_value(value));
    }

	return visit_if_enabled<Feature::index_masks>(
		// Mask can't be const because of masked_view iterator.
		[](auto& array, auto& mask, const auto& value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;
			using VTValue = typename std::decay_t<decltype(value)>::value_type;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else if constexpr (!std::is_convertible_v<VTValue, VTArray>) {
				throw std::runtime_error("Cannot promote this way.");
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
}

void va::set_at_mask(VData& varray, VData& mask, VScalar value) {
	return visit_if_enabled<Feature::index_masks>(
		// Mask can't be const because of masked_view iterator.
		[](auto& array, auto& mask, const auto value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTMask = typename std::decay_t<decltype(mask)>::value_type;
			using VTValue = std::decay_t<decltype(value)>;

			if constexpr (!std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be boolean dtype");
			}
			else if constexpr (!std::is_convertible_v<VTValue, VTArray>) {
				throw std::runtime_error("Cannot promote this way.");
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

std::shared_ptr<VArray> va::get_at_indices(VStoreAllocator& allocator, const VData& data, const VData& indices) {
	return visit_if_enabled<Feature::index_lists>(
		[&allocator, &data](const auto& indices) -> std::shared_ptr<VArray> {
			using VTIndices = typename std::decay_t<decltype(indices)>::value_type;

			if constexpr (!std::is_integral_v<VTIndices> || std::is_same_v<VTIndices, bool>) {
				throw std::runtime_error("mask must be integer dtype");
			}
			else {
				// Index views need to be vectors of xindex.
				xt::svector<xt::svector<size_type>> xindices = array_to_indices(indices);

				return std::visit([&indices, &xindices, &allocator](const auto& array) -> std::shared_ptr<VArray> {
					using VTArray = typename std::decay_t<decltype(array)>::value_type;

					if (indices.dimension() == 1) {
						if (array.dimension() != 1) throw std::runtime_error("cannot use 1D index list for nd tensor");

						return va::create_varray<VTArray>(allocator, xt::index_view(array, xindices));
					}
					if (indices.dimension() != 2) throw std::runtime_error("index list must be 1d or 2d");
					if (indices.shape()[1] != array.dimension()) throw std::runtime_error("index list dimension 2 must match array dimension");

					return va::create_varray<VTArray>(allocator, xt::index_view(array, xindices));
				}, data);
			}
		}, indices
	);
}

void va::set_at_indices(VData& varray, VData& indices, VData& value) {
	visit_if_enabled<Feature::index_lists>(
		[](auto& array, const auto& indices, const auto& value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTMask = typename std::decay_t<decltype(indices)>::value_type;
			using VTValue = typename std::decay_t<decltype(value)>::value_type;

			if constexpr (!std::is_integral_v<VTMask> || std::is_same_v<VTMask, bool>) {
				throw std::runtime_error("mask must be integer dtype");
			}
			else if constexpr (!std::is_convertible_v<VTValue, VTArray>) {
				throw std::runtime_error("Cannot promote this way.");
			}
			else {
				if (indices.dimension() == 1) {
					if (array.dimension() != 1) throw std::runtime_error("cannot use 1D index list for nd tensor");

					auto index_view = xt::index_view(array, indices);

#ifdef XTENSOR_USE_XSIMD
					if constexpr (xtl::is_complex<VTArray>::value) {
						// See above; xsimd cannot auto-convert to complex types
						index_view = xt::cast<VTArray>(value);
					}
					else
#endif
					{
						index_view = value;
					}
					return;
				}
				if (indices.dimension() != 2) throw std::runtime_error("index list must be 1d or 2d");
				if (indices.shape()[1] != array.dimension()) throw std::runtime_error("index list dimension 2 must match array dimension");

				// Index views need to be vectors of xindex.
				xt::svector<xt::svector<size_type>> xindices = array_to_indices(indices);
				auto index_view = xt::index_view(array, xindices);

#ifdef XTENSOR_USE_XSIMD
				if constexpr (xtl::is_complex<VTArray>::value) {
					// See above; xsimd cannot auto-convert to complex types
					index_view = xt::cast<VTArray>(value);
				}
				else
#endif
				{
					index_view = value;
				}
			}
		}, varray, indices, value
	);
}
