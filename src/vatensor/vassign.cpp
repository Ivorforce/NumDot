#include "vassign.hpp"

#include <type_traits>                                  // for decay_t
#include <variant>                                      // for visit
#include "varray.hpp"                            // for VData, VScalar
#include "create.hpp"
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xindex_view.hpp>
#include "vcarray.hpp"
#include "vcompute.hpp"
#include "vpromote.hpp"
#include "xscalar_store.hpp"
#include "vfunc/entrypoints.hpp"

using namespace va;

static void mod_index(axes_type& index, const shape_type& shape) {
	// xtensor actually checks later, too, but it just pads with 0 rather than throwing.
	if (index.size() != shape.size()) throw std::runtime_error("invalid dimension for index");

	for (int i = 0; i < index.size(); ++i) {
		if (index[i] < 0) index[i] += shape[i];
		if (index[i] < 0 || index[i] >= shape[i]) throw std::runtime_error("index out of bounds");
	}
}

void va::set_single_value(VData& array, axes_type& index, const VScalar& value) {
	std::visit(
		[&index](auto& carray, const auto& value) -> void {
			using VWrite = typename std::decay_t<decltype(carray)>::value_type;
			mod_index(index, carray.shape());
			carray[index] = static_cast_scalar<VWrite>(value);
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

void va::assign(VStoreAllocator& allocator, const VArrayTarget& target, const VData& value) {
	if (const auto target_data = std::get_if<VData*>(&target)) {
		VData& data = **target_data;
		va::assign(data, value);
	}
	else {
		auto& target_varray = *std::get<std::shared_ptr<VArray>*>(target);
		target_varray = va::copy(allocator, value);
	}
}

void va::assign(const VArrayTarget& target, VScalar value) {
	if (const auto target_data = std::get_if<VData*>(&target)) {
		VData& data = **target_data;
		va::fill(data, value);
	}
	else {
		auto& target_varray = *std::get<std::shared_ptr<VArray>*>(target);
		target_varray = va::store::from_scalar_variant(value);
	}
}

std::shared_ptr<VArray> va::get_at_mask(VStoreAllocator& allocator, const VData& data, const VData& mask) {
	if (va::dtype(mask) != va::DType::Bool) throw std::runtime_error("mask must be boolean dtype");

	auto& mask_ = std::get<compute_case<bool*>>(mask);
	const size_t array_size = xt::sum(mask_, xt::evaluation_strategy::immediate)();

	auto result_varray = va::empty(allocator, va::dtype(data), shape_type { array_size });
	auto& result_data = result_varray->data;

	std::visit([&mask_, &result_data](const auto& array) {
		using VTArray = typename std::decay_t<decltype(array)>::value_type;

		auto result_compute = std::get<compute_case<VTArray*>>(result_data);

		// Masked views don't offer this functionality automatically.
		const auto masked_view = xt::masked_view(array, mask_);

		auto iter_result = result_compute.begin();
		for (auto masked_value : masked_view) {
			if (masked_value.visible()) {
				*iter_result = masked_value.value();
				++iter_result;
			}
		}
	}, data);

	return result_varray;
}

void va::set_at_mask(VData& varray, VData& mask, VData& value) {
	// This case is not handled again later, so we actually need this 'performance' check.
    if (va::dimension(value) == 0) {
	    return set_at_mask(varray, mask, va::to_single_value(value));
    }

	if (va::dtype(mask) != va::DType::Bool) throw std::runtime_error("mask must be boolean dtype");

	auto& mask_ = std::get<compute_case<bool*>>(mask);
	const size_t array_size = xt::sum(mask_, xt::evaluation_strategy::immediate)();
	const auto& array_shape = va::shape(varray);

	if (array_shape != mask_.shape()) {
		throw std::runtime_error("mask must be same shape as array");
	}

	// Masked views don't offer array fill functionality automatically.
	if (va::shape(value) != shape_type { array_size }) {
		throw std::runtime_error("mask must be single value or match the mask sum");
	}

	std::visit(
		// Mask can't be const because of masked_view iterator.
		[&mask_](auto& array, const auto& value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTValue = typename std::decay_t<decltype(value)>::value_type;

			if constexpr (!std::is_convertible_v<VTValue, VTArray>) {
				throw std::runtime_error("Cannot promote this way.");
			}
			else {
				auto masked_view = xt::masked_view(array, mask_);

				const auto stride = value.strides()[0];
				auto iter_value = value.begin();
				for (auto masked_value : masked_view) {
					if (masked_value.visible()) {
						masked_value.value() = static_cast<VTArray>(*iter_value);
						iter_value += stride;
					}
				}
			}
		}, varray, value
	);
}

void va::set_at_mask(VData& varray, VData& mask, VScalar value) {
	if (va::dtype(mask) != va::DType::Bool) throw std::runtime_error("mask must be boolean dtype");

	auto& mask_ = std::get<compute_case<bool*>>(mask);
	const auto& array_shape = va::shape(varray);

	if (array_shape != mask_.shape()) {
		throw std::runtime_error("mask must be same shape as array");
	}

	std::visit(
		// Mask can't be const because of masked_view iterator.
		[&mask_](auto& array, const auto value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTValue = std::decay_t<decltype(value)>;

			if constexpr (!std::is_convertible_v<VTValue, VTArray>) {
				throw std::runtime_error("Cannot promote this way.");
			}
			else {
				auto masked_view = xt::masked_view(array, mask_);
				masked_view = value;
			}
		}, varray, value
	);
}

template<typename A>
xt::svector<xt::svector<size_type>> array_to_indices(const A& indices) {
	// TODO we could optimize 1d indices, xtensor supports this.
	if (indices.dimension() != 1 && indices.dimension() != 2) throw std::runtime_error("index list must be 1d or 2d");

	const auto num_indices = indices.shape()[0];
	const auto num_dimensions = indices.dimension() == 2 ? indices.shape()[1] : 1;
	const auto stride = indices.strides()[0];

	// TODO This should be possible without allocating separately for each xidx, no?
	xt::svector<xt::svector<size_type>> xindices(num_indices);
	for (int i = 0; i < num_indices; ++i) {
		xt::svector<size_type> xidx(num_dimensions);
		std::copy_n(indices.begin() + i * stride, num_dimensions, xidx.begin());
		xindices[i] = std::move(xidx);
	}
	return xindices;
}

xt::svector<xt::svector<size_type>> get_as_indices(const VData& indices) {
	return std::visit(
		[](const auto& indices) -> xt::svector<xt::svector<size_type>> {
			using VTIndices = typename std::decay_t<decltype(indices)>::value_type;

			if constexpr (!std::is_integral_v<VTIndices> || std::is_same_v<VTIndices, bool>) {
				throw std::runtime_error("mask must be integer dtype");
			}
			else {
				// Index views need to be vectors of xindex.
				return array_to_indices(indices);
			}
		}, indices
	);
}

std::shared_ptr<VArray> va::get_at_indices(VStoreAllocator& allocator, const VData& data, const VData& indices) {
	const auto indices_norm = get_as_indices(indices);
	const auto array_dimension = va::dimension(data);
	const auto& indices_shape = va::shape(indices);

	if (indices_shape.size() == 1 && array_dimension != 1) throw std::runtime_error("cannot use 1D index list for nd tensor");
	if (va::shape(indices)[1] != array_dimension) throw std::runtime_error("index list dimension 2 must match array dimension");

	return std::visit([&indices_norm, &allocator](const auto& array) -> std::shared_ptr<VArray> {
		using VTArray = typename std::decay_t<decltype(array)>::value_type;
		return va::create_varray<VTArray>(allocator, xt::index_view(array, indices_norm));
	}, data);
}

void va::set_at_indices(VData& data, VData& indices, VData& value) {
	const auto indices_norm = get_as_indices(indices);
	const auto array_dimension = va::dimension(data);
	const auto& indices_shape = va::shape(indices);

	if (indices_shape.size() == 1 && array_dimension != 1) throw std::runtime_error("cannot use 1D index list for nd tensor");
	if (indices_shape[1] != array_dimension) throw std::runtime_error("index list dimension 2 must match array dimension");

	std::visit(
		[&indices_norm](auto& array, const auto& value) {
			using VTArray = typename std::decay_t<decltype(array)>::value_type;
			using VTValue = typename std::decay_t<decltype(value)>::value_type;

			if constexpr (!std::is_convertible_v<VTValue, VTArray>) {
				throw std::runtime_error("Cannot promote this way.");
			}
			else {

				// Index views need to be vectors of xindex.
				auto index_view = xt::index_view(array, indices_norm);

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
		}, data, value
	);
}
