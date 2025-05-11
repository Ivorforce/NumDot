//
// Created by Lukas Tenbrink on 12.09.24.
//

#include "rearrange.hpp"

#include <algorithm>                  // for stable_partition
#include <functional>                 // for multiplies
#include <numeric>                    // for accumulate, iota
#include <set>                        // for operator==, set

#include "create.hpp"
#include "util.hpp"
#include "vpromote.hpp"
#include "xscalar_store.hpp"
#include "dtype.hpp"
#include "vatensor//varray.hpp"         // for VArray, strides_type, axes_type
#include "xtensor/core/xlayout.hpp"        // for layout_type
#include "xtensor/misc/xmanipulation.hpp"  // for full, transpose, flip, moveaxis
#include "xtensor/views/xstrided_view.hpp"  // for reshape_view

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

VData va::moveaxis(const VData& data, std::ptrdiff_t src, std::ptrdiff_t dst) {
	return map_compute(
		[src, dst](auto& array) {
			return xt::moveaxis(array, src, dst);
		}, data
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

std::shared_ptr<VArray> va::join_axes_into_last_dimension(const VData& varray, axes_type axes) {
	throw std::runtime_error("join_axes_into_last_dimension not implemented");
	// const auto reduction_count = axes.size();
	//
	// if (reduction_count == 0) {
	// 	return std::make_shared<VArray>(varray);
	// }
	//
	// auto permutation = axes_type(varray.dimension());
	//
	// std::iota(permutation.begin(), permutation.end(), 0);
	// move_indices_to_back(permutation, axes);
	//
	// return map(
	// 	[permutation, reduction_count](auto& carray) {
	// 		auto transposed = xt::transpose(
	// 			carray,
	// 			permutation,
	// 			xt::check_policy::full {}
	// 		);
	// 		shape_type new_shape = transposed.shape();
	// 		auto reduction_begin = new_shape.end() - reduction_count;
	// 		*reduction_begin = std::accumulate(reduction_begin, new_shape.end(), static_cast<std::size_t>(1), std::multiplies());
	// 		new_shape.erase(reduction_begin + 1, new_shape.end());
	// 		return xt::reshape_view(transposed, new_shape);
	// 	}, varray
	// );
}

std::shared_ptr<VArray> reinterpret_complex_as_floats(const std::shared_ptr<VArray>& varray, std::ptrdiff_t offset, bool add_dimension) {
	return std::visit(
	[&varray, offset, add_dimension](auto& carray) -> std::shared_ptr<VArray> {
			using V = typename std::decay_t<decltype(carray)>::value_type;

			if constexpr (xtl::is_complex<V>::value) {
				using V = typename std::decay_t<decltype(carray)>::value_type;

				strides_type new_strides = carray.strides();
				for (auto& stride : new_strides) { stride *= 2; }
				if (add_dimension) new_strides.push_back(1);

				shape_type new_shape = carray.shape();
				if (add_dimension) new_shape.push_back(2);

				return std::make_shared<VArray>(VArray {
					std::shared_ptr(varray->store),
					make_compute(
						reinterpret_cast<typename V::value_type*>(const_cast<V*>(carray.data())) + offset,
						new_shape,
						new_strides,
						(add_dimension && (carray.layout() == xt::layout_type::row_major || carray.layout() == xt::layout_type::any)) ? xt::layout_type::row_major : xt::layout_type::dynamic
					),
					varray->data_offset * 2 + offset
				});
			}
			else {
				return varray;
			}
		}, varray->data
	);
}

std::shared_ptr<VArray> va::real(const std::shared_ptr<VArray>& varray) {
	return reinterpret_complex_as_floats(varray, 0, false);
}

std::shared_ptr<VArray> va::imag(const std::shared_ptr<VArray>& varray) {
	return reinterpret_complex_as_floats(varray, 1, false);
}

std::shared_ptr<VArray> va::complex_as_vector(const std::shared_ptr<VArray>& varray) {
	return reinterpret_complex_as_floats(varray, 0, true);
}

std::shared_ptr<VArray> va::vector_as_complex(VStoreAllocator& allocator, const VArray& varray, DType dtype, bool keepdims) {
	const auto dim_count = varray.dimension();

	if (dim_count < 1) { throw std::invalid_argument("Array must have at least one dimension"); }

	const auto& strides = varray.strides();
	const auto& shape = varray.shape();

	if (shape.back() != 2) { throw std::invalid_argument("Last dimension shape must be 2"); }

	if (strides.back() == 1 && std::visit([dtype](auto& carray) -> bool {
		using V = typename std::decay_t<decltype(carray)>::value_type;

		if constexpr (xtl::is_complex<V>::value) {
			throw std::runtime_error("Complex vector cannot be reinterpreted as real vector");
		}
		else if constexpr (std::is_floating_point_v<V>) {
			return dtype == DTypeMax || dtype == dtype_of_type<std::complex<V>>();
		}
		else {
			return false;
		}
	}, varray.data)) {
		// Can return a view!

		// Remove last dimension.
		auto new_strides = strides_type(dim_count - (keepdims ? 0 : 1));
		for (int i = 0; i < dim_count - 1; ++i) { new_strides[i] = strides[i] / 2; }
		if (keepdims) { new_strides.back() = 0; }

		auto new_shape = shape_type(dim_count - 1);
		std::copy_n(shape.begin(), dim_count - 1, new_shape.begin());
		if (keepdims) { new_shape.back() = 1; }

		const auto new_layout = keepdims ? xt::layout_type::dynamic : varray.layout();

		return std::make_shared<VArray>(VArray {
			std::shared_ptr(varray.store),
			std::visit([&new_shape, &new_strides, &new_layout](auto& carray) -> VData {
				using V = typename std::decay_t<decltype(carray)>::value_type;

				if constexpr (!std::is_floating_point_v<V>) {
					throw std::runtime_error("internal error");
				}
				else {
					return make_compute(
						reinterpret_cast<std::complex<V>*>(const_cast<V*>(carray.data())),
						new_shape,
						new_strides,
						new_layout
					);
				}
			}, varray.data),
			varray.data_offset * 2
		});
	}

	// Need to return a copy.
	if (dtype == DTypeMax) { dtype = DType::Complex128; }

	const DType comp_dtype = complex_dtype_value_type(dtype);

	const auto float_array = va::copy_as_dtype(allocator, varray.data, comp_dtype);
	// Call ourselves again, though this time we should get a view for sure.
	return vector_as_complex(allocator, *float_array, dtype, keepdims);
}

std::shared_ptr<VArray> va::squeeze(const std::shared_ptr<VArray>& varray) {
	xt::xstrided_slice_vector v;
	const shape_type &shape = varray->shape();
	if (std::ranges::count(shape, 1) == 0) {
		return varray;
	}

	// Not the most efficient way, but eh.
	v.resize(shape.size());
	for (int i = 0; i < shape.size(); ++i) {
		v[i] = shape[i] == 1 ? xt::xstrided_slice<std::ptrdiff_t>(0) : xt::xstrided_slice<std::ptrdiff_t>(xt::xall_tag{});
	}

	return varray->sliced(v);
}
