#include "create.hpp"

#include <utility>                      // for move
#include <variant>                      // for visit
#include "rearrange.hpp"
#include "varray.hpp"            // for VArray, shape_type, DType
#include "vassign.hpp"
#include "vcall.hpp"
#include "vcompute.hpp"
#include "vpromote.hpp"
#include "vcarray.hpp"
#include "xtensor/generators/xbuilder.hpp"         // for empty
#include "xtensor/core/xlayout.hpp"          // for layout_type
#include "xtensor/core/xoperation.hpp"       // for cast

using namespace va;

std::shared_ptr<VArray> va::full(VStoreAllocator& allocator, const VScalar fill_value, const shape_type& shape) {
	// Technically for this function we could use va::create_varray, but it's way faster to store the array contiguously.

	auto count = xt::compute_size(shape);
	auto store = allocator.allocate(va::dtype(fill_value), count);
	auto ptr = store->data();

	return std::visit(
		[ptr, count, &shape, &store](auto fill_value) -> std::shared_ptr<VArray> {
			using T = std::decay_t<decltype(fill_value)>;

			std::fill_n(static_cast<T*>(ptr), count, fill_value);

			auto data = make_compute<T*>(
				static_cast<T*>(ptr),
				shape,
				strides_type{}, // unused
				shape.size() <= 1 ? xt::layout_type::any : xt::layout_type::row_major
			);

			return std::make_shared<VArray>(
				VArray {
					std::move(store),
					std::move(data),
					0
				}
			);
		}, fill_value
	);
}

std::shared_ptr<VArray> va::empty(VStoreAllocator& allocator, DType dtype, const shape_type& shape) {
	// Technically for this function we could use va::create_varray, but it's way faster to store the array contiguously.

	auto count = xt::compute_size(shape);
	auto store = allocator.allocate(dtype, count);
	auto ptr = store->data();

	return std::visit(
		[ptr, &shape, &store](auto t) -> std::shared_ptr<VArray> {
			using T = std::decay_t<decltype(t)>;

			auto data = make_compute<T*>(
				static_cast<T*>(ptr),
				shape,
				strides_type{}, // unused
				shape.size() <= 1 ? xt::layout_type::any : xt::layout_type::row_major
			);

			return std::make_shared<VArray>(
				VArray {
					std::move(store),
					std::move(data),
					0
				}
			);
		}, dtype_to_variant(dtype)
	);
}

std::shared_ptr<VArray> va::eye(VStoreAllocator& allocator, DType dtype, const shape_type& shape, int k) {
	// For some reason, xt::eye wants this specific type
	auto shape_eye = std::vector<std::size_t>(shape.size());
	std::copy_n(shape.begin(), shape.size(), shape_eye.begin());

	return std::visit(
		[&shape_eye, &allocator, k](auto t) -> std::shared_ptr<VArray> {
			using T = std::decay_t<decltype(t)>;
			return va::create_varray<T>(allocator, xt::eye<T>(shape_eye, k));
		}, dtype_to_variant(dtype)
	);
}

std::shared_ptr<VArray> va::copy(VStoreAllocator& allocator, const VData& other) {
	auto array = empty(allocator, va::dtype(other), va::shape(other));
	va::assign(array->data, other);
	return array;
}

std::shared_ptr<VArray> va::copy_as_dtype(VStoreAllocator& allocator, const VData& other, DType dtype) {
	if (dtype == DTypeMax) dtype = va::dtype(other);

	auto array = empty(allocator, dtype, va::shape(other));
	va::assign(array->data, other);
	return array;
}

std::shared_ptr<VArray> va::linspace(VStoreAllocator& allocator, VScalar start, VScalar stop, std::size_t num, bool endpoint, DType dtype) {
	if (dtype == DTypeMax) {
		// linspace is always at least float32.
		dtype = DType::Float32;
		dtype = va::dtype_common_type_unchecked(dtype, va::dtype(start));
		dtype = va::dtype_common_type_unchecked(dtype, va::dtype(stop));
	}

	double start_ = static_cast_scalar<double>(start);
	double stop_ = static_cast_scalar<double>(stop);

	double step = 0.0;
	if (num > 0) {
		step = (stop_ - start_) / (static_cast<double>(num) - (endpoint ? 1.0 : 0.0));
	}

	auto array = empty(allocator, dtype, shape_type {num});
	_call_vfunc_inplace<double, double, std::size_t>(va::vfunc::tables::fill_consecutive, array->data, std::move(start_), std::move(step), std::move(num));
	return array;
}

std::shared_ptr<VArray> va::arange(VStoreAllocator& allocator, VScalar start, VScalar stop, VScalar step, DType dtype) {
	if (dtype == DTypeMax) {
		dtype = va::dtype(start);
		dtype = va::dtype_common_type(dtype, va::dtype(stop));
		dtype = va::dtype_common_type(dtype, va::dtype(step));
	}

	std::size_t num;
	if (dtype == Complex64 || dtype == Complex128) {
		throw std::invalid_argument("arange cannot be used with this dtype");
	}
	else if (dtype == Float32 || dtype == Float64) {
		// float-like
		auto start_ = static_cast_scalar<double>(start);
		auto stop_ = static_cast_scalar<double>(stop);
		auto step_ = static_cast_scalar<double>(step);

		// From arange_impl
		num = std::ceil((stop_ - start_) / step_);
	}
	else if (dtype == Int8 || dtype == Int16 || dtype == Int32 || dtype == Int64) {
		// signed int-like
		auto start_ = static_cast_scalar<int64_t>(start);
		auto stop_ = static_cast_scalar<int64_t>(stop);
		auto step_ = static_cast_scalar<int64_t>(step);

		// From arange_impl
		num = (stop_ - start_) / step_;
	}
	else {
		// unsigned int-like
		auto start_ = static_cast_scalar<uint64_t>(start);
		auto stop_ = static_cast_scalar<uint64_t>(stop);
		auto step_ = static_cast_scalar<uint64_t>(step);
		// From arange_impl
		num = (stop_ - start_) / step_;
	}

	start = static_cast_scalar(start, dtype);
	step = static_cast_scalar(step, dtype);

	auto array = empty(allocator, dtype, shape_type {num});
	_call_vfunc_inplace<void*, void*, std::size_t>(va::vfunc::tables::fill_consecutive, array->data, va::_call::get_value_ptr(start), va::_call::get_value_ptr(step), std::move(num));
	return array;
}

std::shared_ptr<VArray> va::tile(VStoreAllocator& allocator, const VArray& array, const shape_type& reps, bool inner) {
	const auto matched_reps = std::min(array.dimension(), reps.size());

	const int shape_shift = inner ? 1 : 2;
	const int repeat_shift = inner ? 2 : 1;

	// We slice the array as many times as necessary.
	// If reps exceeds the array size, array will be auto-broadcast.
	// If array exceeds the rep size, it will match the final shape without needing to be sliced.
	xt::xstrided_slice_vector array_slices(1 + matched_reps * 2);
	array_slices[0] = xt::ellipsis();
	for (int i = 0; i < matched_reps; ++i) {
		array_slices[repeat_shift + i * 2] = xt::newaxis();
		array_slices[shape_shift + i * 2] = xt::all();
	}

	va::shape_type result_final_shape(std::max(array.dimension(), reps.size()));
	va::strides_type result_broadcast_shape(array.dimension() + reps.size());
	int steps_from_end = 0;

	// Fill everything that both array and reps cover.
	for (int i = 0; i < matched_reps; ++i) {
		const std::size_t rep_count = reps[reps.size() - steps_from_end - 1];
		const std::size_t array_dim_size = array.shape()[(array.dimension() - steps_from_end - 1)];

		result_broadcast_shape[result_broadcast_shape.size() - steps_from_end * 2 - shape_shift] = rep_count;
		result_broadcast_shape[result_broadcast_shape.size() - steps_from_end * 2 - repeat_shift] = array_dim_size;

		result_final_shape[result_final_shape.size() - steps_from_end - 1] = rep_count * array_dim_size;

		steps_from_end++;
	}
	// If needed, fill up the rest of the array shape.
	if (array.dimension() > reps.size()) {
		for (std::size_t i = 0; i < array.dimension() - reps.size(); ++i) {
			result_broadcast_shape[i] = array.shape()[i];
			result_final_shape[i] = array.shape()[i];
		}
	}
	// If needed, fill up the rest of the reps shape.
	if (reps.size() > array.dimension()) {
		for (std::size_t i = 0; i < reps.size() - array.dimension(); ++i) {
			result_broadcast_shape[i] = reps[i];
			result_final_shape[i] = reps[i];
		}
	}

	auto result = va::empty(allocator, array.dtype(), result_final_shape);
	auto result_broadcast = va::reshape(allocator, result, result_broadcast_shape);
	const auto array_broadcast = array.sliced_data(array_slices);

	result_broadcast->prepare_write();
	va::assign(result_broadcast->data, array_broadcast);

	return result;
}

std::shared_ptr<VArray> va::reshape(VStoreAllocator& allocator, const std::shared_ptr<VArray>& varray, strides_type new_shape) {
	if (varray->layout() == xt::layout_type::row_major) {
		// Do in-place reshape.
		return map(
			[&new_shape](auto& array) {
				auto new_shape_ = new_shape;
				return xt::reshape_view(array, new_shape_);
			}, *varray
		);
	}

	// Need to allocate new memory for reshape.
	const auto count = varray->size();
	auto store = allocator.allocate(varray->dtype(), count);

	return std::visit(
		[&store, &new_shape](const auto& read) -> std::shared_ptr<VArray> {
			using VT = typename std::decay_t<decltype(read)>::value_type;

			auto ptr = static_cast<VT*>(store->data());
			util::fill_c_array_flat(ptr, read);

			// Use xtensor to figure out the final shape for us.
			// (new_shape can have one negative entry, for "figure out the rest")
			const auto reshape_view = xt::reshape_view(read, new_shape);

			return std::make_shared<VArray>(
				VArray {
					store,
					make_compute<VT*>(
						std::move(ptr),
						reshape_view.shape(),
						strides_type {}, // unused
						xt::layout_type::row_major
					),
					0
				}
			);
		}, varray->data
	);
}

std::shared_ptr<VArray> va::flatten(VStoreAllocator& allocator, const std::shared_ptr<VArray>& varray) {
	if (varray->dimension() == 1) {
		// Fast lane return.
		return varray;
	}

	return reshape(allocator, varray, strides_type { static_cast<std::ptrdiff_t>(varray->size()) });
}
