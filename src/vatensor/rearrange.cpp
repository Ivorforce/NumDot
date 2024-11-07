//
// Created by Lukas Tenbrink on 12.09.24.
//

#include "rearrange.hpp"

#include <algorithm>                  // for stable_partition
#include <functional>                 // for multiplies
#include <numeric>                    // for accumulate, iota
#include <set>                        // for operator==, set
#include "util.hpp"
#include "vpromote.hpp"
#include "xscalar_store.hpp"
#include "vatensor//varray.hpp"         // for VArray, strides_type, axes_type
#include "xtensor/xlayout.hpp"        // for layout_type
#include "xtensor/xmanipulation.hpp"  // for full, transpose, flip, moveaxis
#include "xtensor/xstrided_view.hpp"  // for reshape_view

using namespace va;

std::shared_ptr<VArray> va::transpose(const VArray& varray, strides_type permutation) {
	return map(
		[permutation](auto& array) {
			return xt::transpose(
				array,
				permutation,
				xt::check_policy::full {}
			);
		}, varray
	);
}

std::shared_ptr<VArray> va::transpose(const VArray& varray) {
	const auto dim = varray.dimension();
	va::axes_type permutation_(dim);
	for (std::size_t i = 0; i < dim; ++i) {
		permutation_[i] = static_cast<int>(dim - 1 - i);
	}

	return va::transpose(varray, permutation_);
}

std::shared_ptr<VArray> va::reshape(const VArray& varray, strides_type new_shape) {
	return map(
		[new_shape](auto& array) {
			auto new_shape_ = new_shape;
			return xt::reshape_view(array, new_shape_);
		}, varray
	);
}

std::shared_ptr<VArray> va::swapaxes(const VArray& varray, std::ptrdiff_t a, std::ptrdiff_t b) {
	return map(
		[a, b](auto& array) {
			return xt::swapaxes(array, a, b);
		}, varray
	);
}

std::shared_ptr<VArray> va::moveaxis(const VArray& varray, std::ptrdiff_t src, std::ptrdiff_t dst) {
	return map(
		[src, dst](auto& array) {
			return xt::moveaxis(array, src, dst);
		}, varray
	);
}

std::shared_ptr<VArray> va::flip(const VArray& varray, std::size_t axis) {
	return map(
		[axis](auto& array) {
			return xt::flip(array, axis);
		}, varray
	);
}

std::shared_ptr<VArray> va::diagonal(const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2) {
	const size_t dimension = varray.dimension();
	if (dimension < 2) {
		throw std::runtime_error("diagonal requires array must have at least two dimensions");
	}

	const auto axis1_ = va::util::normalize_axis(axis1, dimension);
	const auto axis2_ = va::util::normalize_axis(axis2, dimension);
	if (axis1_ == axis2_) {
		throw std::runtime_error("axes cannot be the same");
	}

	std::size_t shape_size_1;
	std::size_t shape_size_2;
	std::ptrdiff_t strides_1;
	std::ptrdiff_t strides_2;

	const shape_type& shape = varray.shape();
	const strides_type& strides = varray.strides();

	shape_type new_shape(dimension - 1);
	strides_type new_strides(dimension - 1);
	for (int i = 0, j = 0; i < dimension; ++i) {
		if (i == axis1_) {
			shape_size_1 = shape[i];
			strides_1 = strides[i];
			continue;
		}
		if (i == axis2_) {
			shape_size_2 = shape[i];
			strides_2 = strides[i];
			continue;
		}
		new_shape[j] = shape[i];
		j++;
	}

	std::ptrdiff_t added_data_offset = 0;

	if (offset < 0) {
		// Move along axis1
		if (shape_size_1 > -offset) {
			shape_size_1 = shape_size_1 - -offset;
			added_data_offset = -offset * strides_1;
		}
		else {
			// We're cooked anyway, offset points us outside the array.
			// Just make the size 0 and keep the data pointer where it is.
			shape_size_1 = 0;
		}
	}
	else if (offset > 0) {
		// Move along axis2
		if (shape_size_2 > offset) {
			shape_size_2 = shape_size_2 - offset;
			added_data_offset = offset * strides_2;
		}
		else {
			// We're cooked anyway, offset points us outside the array.
			// Just make the size 0 and keep the data pointer where it is.
			shape_size_2 = 0;
		}
	}

	// New shape is whatever both can provide.
	new_shape[dimension - 2] = std::min(shape_size_1, shape_size_2);
	// Strides is both strides at once.
	new_strides[dimension - 2] = strides_1 + strides_2;

	return std::visit(
		[&varray, &new_shape, &new_strides, added_data_offset](const auto& read) -> std::shared_ptr<VArray> {
			using VT = typename std::decay_t<decltype(read)>::value_type;

			return std::make_shared<VArray>(
				VArray {
					std::shared_ptr(varray.store),
					make_compute<VT*>(
						const_cast<VT*>(read.data()) + added_data_offset,
						new_shape,
						new_strides,
						xt::layout_type::dynamic
					),
					varray.data_offset + added_data_offset
				}
			);
		}, varray.data
	);
}

template<typename T, typename I>
void move_indices_to_back(T& vec, const I& indices) {
	using ValueType = typename T::value_type;
	std::set<ValueType> indexSet(indices.begin(), indices.end());

	std::stable_partition(
		vec.begin(), vec.end(), [&indexSet](const ValueType& value) {
			// .contains is C++20
			return indexSet.find(value) == indexSet.end();
		}
	);
}

std::shared_ptr<VArray> va::join_axes_into_last_dimension(const VArray& varray, axes_type axes) {
	const auto reduction_count = axes.size();

	if (reduction_count == 0) {
		return std::make_shared<VArray>(varray);
	}

	auto permutation = axes_type(varray.dimension());

	std::iota(permutation.begin(), permutation.end(), 0);
	move_indices_to_back(permutation, axes);

	return map(
		[permutation, reduction_count](auto& carray) {
			auto transposed = xt::transpose(
				carray,
				permutation,
				xt::check_policy::full {}
			);
			shape_type new_shape = transposed.shape();
			auto reduction_begin = new_shape.end() - reduction_count;
			*reduction_begin = std::accumulate(reduction_begin, new_shape.end(), static_cast<std::size_t>(1), std::multiplies());
			new_shape.erase(reduction_begin + 1, new_shape.end());
			return xt::reshape_view(transposed, new_shape);
		}, varray
	);
}

template <typename T>
std::shared_ptr<VArray> reinterpret_complex_as_floats(const VArray& varray, const T& carray, std::ptrdiff_t offset) {
    using V = typename std::decay_t<decltype(carray)>::value_type;

	auto new_strides = carray.strides();
	for (auto& stride : new_strides) { stride *= 2; }

	return std::make_shared<VArray>(VArray {
		std::shared_ptr(varray.store),
		make_compute(
			reinterpret_cast<typename V::value_type*>(const_cast<V*>(carray.data())) + offset,
			carray.shape(),
			new_strides,
			xt::layout_type::dynamic
		),
		varray.data_offset * 2 + offset
	});
}

std::shared_ptr<VArray> va::real(const std::shared_ptr<VArray>& varray) {
	return std::visit(
		[&varray](auto& carray) -> std::shared_ptr<VArray> {
		    using V = typename std::decay_t<decltype(carray)>::value_type;

			if constexpr (xtl::is_complex<V>::value) {
				return reinterpret_complex_as_floats(*varray, carray, 0);
			}
			else {
				return varray;
			}
		}, varray->data
	);
}

std::shared_ptr<VArray> va::imag(const std::shared_ptr<VArray>& varray) {
	return std::visit(
		[&varray](auto& carray) -> std::shared_ptr<VArray> {
			using V = typename std::decay_t<decltype(carray)>::value_type;
			if constexpr (xtl::is_complex<V>::value) {
				return reinterpret_complex_as_floats(*varray, carray, 1);
			}
			else {
				return va::store::full_dummy_like(0, carray);
			}
		}, varray->data
	);
}
