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
#include "vassign.hpp"
#include "vcall.hpp"
#include "vpromote.hpp"
#include "xscalar_store.hpp"
#include "dtype.hpp"
#include "vatensor//varray.hpp"         // for VArray, strides_type, axes_type
#include "xtensor/core/xlayout.hpp"        // for layout_type
#include "xtensor/misc/xmanipulation.hpp"  // for full, transpose, flip, moveaxis
#include "xtensor/misc/xsort.hpp"          // for argmax, argmin
#include "xtensor/core/xoperation.hpp"     // for nonzero
#include "xtensor/views/xstrided_view.hpp"  // for reshape_view

using namespace va;

std::shared_ptr<VArray> va::transpose(const VArray& varray, strides_type permutation) {
	// xtensor's full check rejects negative axes; the Array API allows them
	// to wrap from the end. Normalize before passing through.
	const auto ndim = static_cast<std::ptrdiff_t>(varray.dimension());
	for (auto& ax : permutation) {
		if (ax < 0) ax += ndim;
	}
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

std::shared_ptr<VArray> va::moveaxis(const VArray& varray, const axes_type& src, const axes_type& dst) {
	if (src.size() != dst.size()) {
		throw std::runtime_error("moveaxis: source and destination must have the same length");
	}
	const std::size_t ndim = varray.dimension();

	std::vector<std::size_t> src_norm(src.size());
	std::vector<std::size_t> dst_norm(dst.size());
	for (std::size_t i = 0; i < src.size(); ++i) {
		src_norm[i] = va::util::normalize_axis(src[i], ndim);
		dst_norm[i] = va::util::normalize_axis(dst[i], ndim);
	}

	std::set<std::size_t> src_set(src_norm.begin(), src_norm.end());
	std::set<std::size_t> dst_set(dst_norm.begin(), dst_norm.end());
	if (src_set.size() != src_norm.size() || dst_set.size() != dst_norm.size()) {
		throw std::runtime_error("moveaxis: repeated axis");
	}

	axes_type order;
	order.reserve(ndim);
	for (std::size_t i = 0; i < ndim; ++i) {
		if (!src_set.contains(i)) order.push_back(static_cast<std::ptrdiff_t>(i));
	}

	// Insert at smallest destination first so later insertions land where requested.
	std::vector<std::pair<std::size_t, std::size_t>> pairs;
	pairs.reserve(src_norm.size());
	for (std::size_t i = 0; i < src_norm.size(); ++i) {
		pairs.emplace_back(dst_norm[i], src_norm[i]);
	}
	std::sort(pairs.begin(), pairs.end());

	for (const auto& [d, s] : pairs) {
		order.insert(order.begin() + d, static_cast<std::ptrdiff_t>(s));
	}

	return va::transpose(varray, order);
}

std::shared_ptr<VArray> va::flip(const VArray& varray, std::ptrdiff_t axis) {
	const auto axis_norm = va::util::normalize_axis(axis, varray.dimension());
	return map(
		[axis_norm](auto& array) {
			return xt::flip(array, axis_norm);
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

std::shared_ptr<VArray> va::roll(VStoreAllocator& allocator, const VArray& varray, std::ptrdiff_t shift, std::ptrdiff_t axis) {
	const std::size_t ndim = varray.dimension();
	const std::size_t axis_norm = va::util::normalize_axis(axis, ndim);
	const std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(varray.shape()[axis_norm]);

	auto out = va::empty(allocator, varray.dtype(), varray.shape());
	if (dim == 0) return out;  // axis has length 0 — nothing to roll

	std::ptrdiff_t k = shift % dim;
	if (k < 0) k += dim;
	if (k == 0) {
		va::assign(out->data, varray.data);
		return out;
	}

	// Roll along one axis is two slice copies:
	//   out[..., 0:k, ...]   ← varray[..., dim-k:dim, ...]
	//   out[..., k:dim, ...] ← varray[..., 0:dim-k, ...]
	xt::xstrided_slice_vector s_in(ndim, xt::all());
	xt::xstrided_slice_vector s_out(ndim, xt::all());

	s_in[axis_norm] = xt::xrange(dim - k, dim);
	s_out[axis_norm] = xt::xrange(static_cast<std::ptrdiff_t>(0), k);
	{
		auto write = out->sliced_data(s_out);
		va::assign(write, varray.sliced_data(s_in));
	}

	s_in[axis_norm] = xt::xrange(static_cast<std::ptrdiff_t>(0), dim - k);
	s_out[axis_norm] = xt::xrange(k, dim);
	{
		auto write = out->sliced_data(s_out);
		va::assign(write, varray.sliced_data(s_in));
	}

	return out;
}

std::shared_ptr<VArray> va::roll(VStoreAllocator& allocator, const VArray& varray, std::ptrdiff_t shift) {
	// Spec: roll without axis flattens, rolls, then reshapes back to input shape.
	const auto self = std::make_shared<VArray>(varray);
	const auto flat = va::flatten(allocator, self);
	auto rolled = va::roll(allocator, *flat, shift, 0);
	if (varray.dimension() == 1) return rolled;
	va::strides_type orig_shape(varray.shape().begin(), varray.shape().end());
	return va::reshape(allocator, rolled, std::move(orig_shape));
}

std::shared_ptr<VArray> va::repeat(VStoreAllocator& allocator, const VArray& varray, const std::vector<std::size_t>& repeats, std::ptrdiff_t axis) {
	const std::size_t axis_norm = va::util::normalize_axis(axis, varray.dimension());
	if (repeats.size() != varray.shape()[axis_norm]) {
		throw std::runtime_error("repeat: per-element repeats length must match the axis size");
	}

	// Output shape: input shape with axis size replaced by sum(repeats).
	const std::size_t total = std::accumulate(repeats.begin(), repeats.end(), std::size_t(0));
	va::shape_type out_shape(varray.shape().begin(), varray.shape().end());
	out_shape[axis_norm] = total;

	return std::visit([&](const auto& in_data) -> std::shared_ptr<VArray> {
		using VT = typename std::decay_t<decltype(in_data)>::value_type;
		auto repeated = xt::repeat(in_data, repeats, axis_norm);

		auto store = allocator.allocate(varray.dtype(), std::accumulate(out_shape.begin(), out_shape.end(), std::size_t(1), std::multiplies<>()));
		auto* ptr = static_cast<VT*>(store->data());
		auto out_compute = make_compute<VT*>(std::move(ptr), out_shape, strides_type{}, xt::layout_type::row_major);

		va::broadcasting_assign_typesafe(out_compute, repeated);

		return std::make_shared<VArray>(VArray { std::move(store), std::move(out_compute), 0 });
	}, varray.data);
}

std::shared_ptr<VArray> va::repeat(VStoreAllocator& allocator, const VArray& varray, std::size_t repeats, std::ptrdiff_t axis) {
	// Broadcast scalar to per-element vector and forward; xt::repeat does the
	// same internally, but going through our vector overload reuses the shape
	// computation and validation logic.
	const std::size_t axis_norm = va::util::normalize_axis(axis, varray.dimension());
	std::vector<std::size_t> per_elem(varray.shape()[axis_norm], repeats);
	return va::repeat(allocator, varray, per_elem, static_cast<std::ptrdiff_t>(axis_norm));
}

std::shared_ptr<VArray> va::expand_dims(const VArray& varray, std::ptrdiff_t axis) {
	const std::size_t ndim = varray.dimension();
	// Array API allows axis in [-ndim-1, ndim]; -ndim-1 means "insert at front".
	const std::ptrdiff_t lo = -static_cast<std::ptrdiff_t>(ndim) - 1;
	const std::ptrdiff_t hi = static_cast<std::ptrdiff_t>(ndim);
	if (axis < lo || axis > hi) {
		throw std::out_of_range("expand_dims: axis out of range");
	}
	const std::size_t pos = axis < 0 ? static_cast<std::size_t>(axis + hi + 1) : static_cast<std::size_t>(axis);
	return map(
		[pos](auto& array) {
			return xt::expand_dims(array, pos);
		}, varray
	);
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

std::shared_ptr<VArray> va::squeeze(const std::shared_ptr<VArray>& varray, const axes_type& axes) {
	const shape_type& shape = varray->shape();
	const std::size_t ndim = shape.size();

	std::set<std::size_t> drop;
	for (const auto a : axes) {
		const auto a_norm = va::util::normalize_axis(a, ndim);
		if (shape[a_norm] != 1) {
			throw std::runtime_error("squeeze: requested axis is not size 1");
		}
		drop.insert(a_norm);
	}

	if (drop.empty()) {
		return varray;
	}

	xt::xstrided_slice_vector v(ndim);
	for (std::size_t i = 0; i < ndim; ++i) {
		v[i] = drop.contains(i)
			? xt::xstrided_slice<std::ptrdiff_t>(0)
			: xt::xstrided_slice<std::ptrdiff_t>(xt::xall_tag{});
	}
	return varray->sliced(v);
}

std::shared_ptr<VArray> va::broadcast_to(const VArray& varray, const shape_type& target_shape) {
	const shape_type& in_shape = varray.shape();
	const strides_type& in_strides = varray.strides();
	const std::size_t in_ndim = in_shape.size();
	const std::size_t out_ndim = target_shape.size();

	if (out_ndim < in_ndim) {
		throw std::runtime_error("broadcast_to: target shape has fewer dimensions than input");
	}

	// Right-align input axes against target. Front-padded axes get stride 0
	// (they replicate the single underlying slice across the new dimension).
	const std::size_t pad = out_ndim - in_ndim;
	strides_type new_strides(out_ndim);
	for (std::size_t i = 0; i < out_ndim; ++i) {
		if (i < pad) {
			new_strides[i] = 0;
			continue;
		}
		const std::size_t in_idx = i - pad;
		const auto in_dim = in_shape[in_idx];
		const auto out_dim = target_shape[i];
		if (in_dim == out_dim) {
			new_strides[i] = in_strides[in_idx];
		}
		else if (in_dim == 1) {
			new_strides[i] = 0;
		}
		else {
			throw std::runtime_error("broadcast_to: input shape not broadcastable to target shape");
		}
	}

	return std::visit(
		[&varray, &target_shape, &new_strides](const auto& read) -> std::shared_ptr<VArray> {
			using VT = typename std::decay_t<decltype(read)>::value_type;
			return std::make_shared<VArray>(
				VArray {
					std::shared_ptr(varray.store),
					make_compute<VT*>(
						const_cast<VT*>(read.data()),
						target_shape,
						new_strides,
						xt::layout_type::dynamic
					),
					varray.data_offset
				}
			);
		}, varray.data
	);
}

namespace {
	// Allocate a fresh int64 VArray of the given shape (row-major).
	std::shared_ptr<VArray> make_int64_array(VStoreAllocator& allocator, const va::shape_type& shape) {
		const std::size_t count = std::accumulate(shape.begin(), shape.end(), std::size_t(1), std::multiplies<>());
		auto store = allocator.allocate(va::Int64, count);
		auto* ptr = static_cast<int64_t*>(store->data());
		auto data = va::make_compute<int64_t*>(std::move(ptr), shape, va::strides_type{}, xt::layout_type::row_major);
		return std::make_shared<VArray>(VArray { std::move(store), std::move(data), 0 });
	}
}

// Complex types have no total order, so xt::argmax/argmin won't instantiate
// for them. Per the array-api spec, argmax/argmin accept only real dtypes,
// so reject complex up front.
namespace {
	template <typename Visitor>
	void visit_real_only(const VData& data, Visitor&& visitor) {
		std::visit([&](const auto& in) {
			using T = typename std::decay_t<decltype(in)>::value_type;
			if constexpr (xtl::is_complex<T>::value) {
				throw std::runtime_error("argmax / argmin not supported for complex dtypes");
			}
			else {
				visitor(in);
			}
		}, data);
	}
}

std::shared_ptr<VArray> va::argmax(VStoreAllocator& allocator, const VArray& varray) {
	auto out = make_int64_array(allocator, va::shape_type{});
	auto& out_compute = std::get<va::compute_case<int64_t*>>(out->data);
	visit_real_only(varray.data, [&out_compute](const auto& in) {
		va::broadcasting_assign_typesafe(out_compute, xt::argmax(in));
	});
	return out;
}

std::shared_ptr<VArray> va::argmax(VStoreAllocator& allocator, const VArray& varray, std::ptrdiff_t axis) {
	const std::size_t axis_norm = va::util::normalize_axis(axis, varray.dimension());
	va::shape_type out_shape;
	out_shape.reserve(varray.dimension() - 1);
	for (std::size_t i = 0; i < varray.dimension(); ++i) {
		if (i != axis_norm) out_shape.push_back(varray.shape()[i]);
	}
	auto out = make_int64_array(allocator, out_shape);
	auto& out_compute = std::get<va::compute_case<int64_t*>>(out->data);
	visit_real_only(varray.data, [&out_compute, axis_norm](const auto& in) {
		va::broadcasting_assign_typesafe(out_compute, xt::argmax(in, static_cast<std::ptrdiff_t>(axis_norm)));
	});
	return out;
}

std::shared_ptr<VArray> va::argmin(VStoreAllocator& allocator, const VArray& varray) {
	auto out = make_int64_array(allocator, va::shape_type{});
	auto& out_compute = std::get<va::compute_case<int64_t*>>(out->data);
	visit_real_only(varray.data, [&out_compute](const auto& in) {
		va::broadcasting_assign_typesafe(out_compute, xt::argmin(in));
	});
	return out;
}

std::shared_ptr<VArray> va::argmin(VStoreAllocator& allocator, const VArray& varray, std::ptrdiff_t axis) {
	const std::size_t axis_norm = va::util::normalize_axis(axis, varray.dimension());
	va::shape_type out_shape;
	out_shape.reserve(varray.dimension() - 1);
	for (std::size_t i = 0; i < varray.dimension(); ++i) {
		if (i != axis_norm) out_shape.push_back(varray.shape()[i]);
	}
	auto out = make_int64_array(allocator, out_shape);
	auto& out_compute = std::get<va::compute_case<int64_t*>>(out->data);
	visit_real_only(varray.data, [&out_compute, axis_norm](const auto& in) {
		va::broadcasting_assign_typesafe(out_compute, xt::argmin(in, static_cast<std::ptrdiff_t>(axis_norm)));
	});
	return out;
}

std::vector<std::shared_ptr<va::VArray>> va::nonzero(VStoreAllocator& allocator, const VArray& varray) {
	const std::size_t ndim = varray.dimension();
	if (ndim == 0) {
		throw std::runtime_error("nonzero is not defined for 0-D arrays");
	}
	// xt::nonzero's element check is `if (x)`, which doesn't compile for
	// complex — fall back to a bool view via the typed visit so the complex
	// branch never instantiates xt::nonzero on a complex array.
	auto indices = std::visit([&](const auto& in) {
		using T = typename std::decay_t<decltype(in)>::value_type;
		if constexpr (xtl::is_complex<T>::value) {
			const auto as_bool = va::copy_as_dtype(allocator, varray.data, va::Bool);
			return xt::nonzero(std::get<va::compute_case<bool*>>(as_bool->data));
		}
		else {
			return xt::nonzero(in);
		}
	}, varray.data);

	std::vector<std::shared_ptr<VArray>> outputs(ndim);
	for (std::size_t d = 0; d < ndim; ++d) {
		const auto& src = indices[d];
		auto out = make_int64_array(allocator, va::shape_type { src.size() });
		auto& out_compute = std::get<va::compute_case<int64_t*>>(out->data);
		std::copy(src.begin(), src.end(), out_compute.begin());
		outputs[d] = std::move(out);
	}
	return outputs;
}
