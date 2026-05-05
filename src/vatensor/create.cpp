#include "create.hpp"

#include <utility>                      // for move
#include <variant>                      // for visit
#include <godot_cpp/core/print_string.hpp>

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

std::shared_ptr<VArray> va::eye(VStoreAllocator& allocator, DType dtype, const shape_type& shape, std::ptrdiff_t k) {
	// xt::eye narrows k to int. Short-circuit |k| values that fall outside the
	// matrix bounds (where the diagonal is empty anyway) so a 64-bit k can't
	// wrap back into a valid offset on the way to xtensor.
	const std::ptrdiff_t rows = !shape.empty() ? static_cast<std::ptrdiff_t>(shape[0]) : 0;
	const std::ptrdiff_t cols = shape.size() > 1 ? static_cast<std::ptrdiff_t>(shape[1]) : rows;
	if (k >= cols || k <= -rows) {
		return va::full(allocator, va::static_cast_scalar(VScalar(int64_t(0)), dtype), shape);
	}

	auto shape_eye = std::vector<std::size_t>(shape.size());
	std::copy_n(shape.begin(), shape.size(), shape_eye.begin());

	return std::visit(
		[&shape_eye, &allocator, k](auto t) -> std::shared_ptr<VArray> {
			using T = std::decay_t<decltype(t)>;
			return va::create_varray<T>(allocator, xt::eye<T>(shape_eye, static_cast<int>(k)));
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

std::shared_ptr<VArray> va::linspace(VStoreAllocator& allocator, VScalar start, VScalar stop, std::size_t num, bool endpoint, DType dtype) {	// Need to process with the highest needed precision. Otherwise, rounding will destroy our values.
	// Need to process with the highest needed precision. Otherwise, rounding will destroy our values.
	// We'll convert to the requested dtype at the end.
	// Use at least float32. That's because we don't know in advance if step needs to be float.
	DType process_dtype = DType::Float32;
	process_dtype = va::dtype_common_type(process_dtype, va::dtype(start));;
	process_dtype = va::dtype_common_type(process_dtype, va::dtype(stop));

	double start_ = static_cast_scalar<double>(start);
	double stop_ = static_cast_scalar<double>(stop);

	double step_ = 0.0;
	// num <= 1 leaves step at 0: out is empty (num=0) or just {start} (num=1).
	// Computing the step would divide by zero (num=1 with endpoint) and bake nan
	// into the single cell, even though numpy returns [start] for that case.
	if (num > 1) {
		step_ = (stop_ - start_) / (static_cast<double>(num) - (endpoint ? 1.0 : 0.0));
	}

	start = static_cast_scalar(start_, process_dtype);
	VScalar step = static_cast_scalar(step_, process_dtype);

	auto array = empty(allocator, process_dtype, shape_type {num});
	_call_vfunc_inplace<const void*, const void*, std::size_t>(va::vfunc::tables::fill_consecutive, array->data, va::_call::get_value_ptr(start), va::_call::get_value_ptr(step), std::move(num));

	// start + step*(num-1) drifts a few ULPs even in double, so the last cell
	// almost never lands on `stop` exactly. Numpy fixes this by overwriting
	// out[-1] = stop after the fill — same trick here.
	if (endpoint && num > 1) {
		va::axes_type last_idx { static_cast<std::ptrdiff_t>(num - 1) };
		va::set_single_value(array->data, last_idx, static_cast_scalar(stop_, process_dtype));
	}

	if (dtype != process_dtype && dtype != DType::DTypeMax) {
		return va::copy_as_dtype(allocator, array->data, dtype);
	}

	return array;
}

std::shared_ptr<VArray> va::arange(VStoreAllocator& allocator, VScalar start, VScalar stop, VScalar step, DType dtype) {
	// Need to process with the highest needed precision. Otherwise, rounding will destroy our values.
	// We'll convert to the requested dtype at the end.
	DType process_dtype = va::dtype(start);
	process_dtype = va::dtype_common_type(process_dtype, va::dtype(stop));
	process_dtype = va::dtype_common_type(process_dtype, va::dtype(step));

	std::size_t num;

	std::visit([&start, &stop, &step, &num](auto t) {
		using T = std::decay_t<decltype(t)>;

		if constexpr (xtl::is_complex<T>::value) {
			throw std::invalid_argument("arange cannot be used with this process_dtype");
		}
		else {
			T start_ = static_cast_scalar<T>(start);
			T stop_ = static_cast_scalar<T>(stop);
			T step_ = static_cast_scalar<T>(step);

			if (step_ == T(0)) {
				throw std::runtime_error("arange: step cannot be zero");
			}

			// numpy semantics: if step's direction disagrees with
			// (stop - start), the result is empty. Guard before computing
			// num so we never assign a negative ceil() to size_t (UB) or
			// suffer signed-overflow inside the difference.
			const bool empty_range = (step_ > T(0)) ? (stop_ <= start_) : (stop_ >= start_);
			if (empty_range) {
				num = 0;
			}
			else if constexpr (std::is_integral_v<T>) {
				// Integer arithmetic preserves the exact difference. Casting
				// to double first would lose ULPs above 2^53 even for tiny
				// ranges (because each operand rounds independently).
				// Overflow of (stop_ - start_) only happens for ranges so
				// large that the subsequent allocation would fail anyway.
				const T diff = stop_ - start_;
				const T q = diff / step_;
				const T r = diff % step_;
				num = static_cast<std::size_t>(q + (r != T(0) ? T(1) : T(0)));
			}
			else {
				num = static_cast<std::size_t>(std::ceil((stop_ - start_) / step_));
			}

			start = start_;
			step = step_;
		}
	}, dtype_to_variant(process_dtype));

	auto array = empty(allocator, process_dtype, shape_type {num});
	_call_vfunc_inplace<const void*, const void*, std::size_t>(va::vfunc::tables::fill_consecutive, array->data, va::_call::get_value_ptr(start), va::_call::get_value_ptr(step), std::move(num));

	if (dtype != process_dtype && dtype != DType::DTypeMax) {
		return va::copy_as_dtype(allocator, array->data, dtype);
	}

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
	if (varray->is_contiguous()) {
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

std::vector<std::shared_ptr<va::VArray>> va::meshgrid(VStoreAllocator& allocator, const std::vector<std::shared_ptr<VArray>>& inputs, bool xy_indexing) {
	const std::size_t n = inputs.size();
	if (n == 0) return {};

	// All inputs must be 1-D; record their lengths.
	va::shape_type lens(n);
	for (std::size_t i = 0; i < n; ++i) {
		if (inputs[i]->dimension() != 1) {
			throw std::runtime_error("meshgrid: each input must be 1-D");
		}
		lens[i] = inputs[i]->shape()[0];
	}

	// "xy" indexing swaps the first two output dims; "ij" leaves them alone.
	// Higher dims are never reordered.
	auto axis_of = [n, xy_indexing](std::size_t i) -> std::size_t {
		if (xy_indexing && n >= 2) {
			if (i == 0) return 1;
			if (i == 1) return 0;
		}
		return i;
	};

	va::shape_type out_shape(n);
	for (std::size_t i = 0; i < n; ++i) out_shape[axis_of(i)] = lens[i];

	std::vector<std::shared_ptr<VArray>> outputs(n);
	for (std::size_t i = 0; i < n; ++i) {
		// Reshape input i to (1, ..., len_i, ..., 1) so broadcast-assign fills
		// each output cell with the right element.
		va::strides_type reshape_dims(n, 1);
		reshape_dims[axis_of(i)] = static_cast<std::ptrdiff_t>(lens[i]);
		const auto reshaped = va::reshape(allocator, inputs[i], reshape_dims);

		auto out = va::empty(allocator, inputs[i]->dtype(), out_shape);
		va::assign(out->data, reshaped->data);
		outputs[i] = out;
	}
	return outputs;
}

std::shared_ptr<VArray> va::flatten(VStoreAllocator& allocator, const std::shared_ptr<VArray>& varray) {
	if (varray->dimension() == 1) {
		// Fast lane return.
		return varray;
	}

	return reshape(allocator, varray, strides_type { static_cast<std::ptrdiff_t>(varray->size()) });
}
