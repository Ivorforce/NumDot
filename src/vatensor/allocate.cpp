#include "allocate.hpp"

#include <utility>                      // for move
#include <variant>                      // for visit

#include "rearrange.hpp"
#include "varray.hpp"            // for VArray, shape_type, DType
#include "vassign.hpp"
#include "vpromote.hpp"
#include "xtensor/xbuilder.hpp"         // for empty
#include "xtensor/xlayout.hpp"          // for layout_type
#include "xtensor/xoperation.hpp"       // for cast

using namespace va;

std::shared_ptr<VArray> empty(VScalar type, const shape_type& shape) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	return std::visit(
		[&shape](auto t) {
			using T = decltype(t);
			return store::from_store(va::array_case<T>(xt::empty<T>(shape)));
		}, type
	);
#endif
}

std::shared_ptr<VArray> va::full(const VScalar fill_value, const shape_type& shape) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	// This is duplicate code, but by filling the store directly instead of the VArray we avoid a few checks, speeding it up a ton.
	return std::visit(
		[&shape](auto fill_value) {
			using T = decltype(fill_value);
			auto store = va::array_case<T>(xt::empty<T>(shape));
			store.fill(fill_value);
			return store::from_store(std::move(store));
		}, fill_value
	);
#endif
}

std::shared_ptr<VArray> va::empty(DType dtype, const shape_type& shape) {
	return ::empty(dtype_to_variant(dtype), shape);
}

std::shared_ptr<VArray> va::eye(DType dtype, const shape_type& shape, int k) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	// For some reason, xt::eye wants this specific type
	auto shape_eye = std::vector<std::size_t>(shape.size());
	std::copy(shape.begin(), shape.end(), shape_eye.begin());
	return std::visit(
		[&shape_eye, k](auto t) -> std::shared_ptr<VArray> {
			using T = decltype(t);
			return store::from_store(store::make_store(xt::eye<T>(shape_eye, k)));
		}, dtype_to_variant(dtype)
	);
#endif
}

std::shared_ptr<VArray> va::copy(const VData& read) {
	return std::visit(
		[](auto& carray) {
			return store::from_store(store::make_store(carray));
		}, read
	);
}

std::shared_ptr<VArray> va::copy_as_dtype(const VData& other, DType dtype) {
#ifdef NUMDOT_DISABLE_ALLOCATION_FUNCTIONS
    throw std::runtime_error("function explicitly disabled; recompile without NUMDOT_DISABLE_ALLOCATION_FUNCTIONS to enable it.");
#else
	if (dtype == DTypeMax) dtype = va::dtype(other);

	return std::visit(
		[](auto t, auto carray) -> std::shared_ptr<VArray> {
			using TWeWanted = decltype(t);
			using TWeGot = typename decltype(carray)::value_type;

			if constexpr (std::is_same_v<TWeWanted, TWeGot>) {
				return store::from_store(va::array_case<TWeWanted>(carray));
			}
			else if constexpr (!std::is_convertible_v<TWeGot, TWeWanted>) {
				throw std::runtime_error("Cannot promote in this way.");
			}
			else {
				// Cast first to reduce number of combinations down the line.
				return store::from_store(va::array_case<TWeWanted>(va::promote::promote_value_type_if_needed_fast<TWeWanted>(carray)));
			}
		}, dtype_to_variant(dtype), other
	);
#endif
}

std::shared_ptr<VArray> va::tile(const VArray& array, const shape_type& reps, bool inner) {
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

		auto result = va::empty(array.dtype(), result_final_shape);
		auto result_broadcast = va::reshape(*result, result_broadcast_shape);
		const auto array_broadcast = array.sliced_data(array_slices);

		result_broadcast->prepare_write();
		va::assign(result_broadcast->data, array_broadcast);

		return result;
}
