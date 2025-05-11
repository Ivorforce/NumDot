#include "stride_tricks.hpp"

#include <variant>
#include "vfunc/entrypoints.hpp"

std::shared_ptr<va::VArray> va::as_strided(const VArray& array, const shape_type& shape, const strides_type& strides) {
	return std::visit(
		[&array, &shape, &strides](const auto& read) -> std::shared_ptr<VArray> {
			using VTRead = typename std::decay_t<decltype(read)>::value_type;

			return std::make_shared<VArray>(
				VArray {
					array.store,
					make_compute<VTRead*>(
						const_cast<VTRead*>(read.data()),
						shape,
						strides,
						xt::layout_type::dynamic
					),
					array.data_offset
				}
			);
		}, array.data
	);
}

std::shared_ptr<va::VArray> va::sliding_window_view(const VArray& array, const shape_type& window_shape) {
	const std::size_t dimension = array.dimension();
	if (window_shape.size() > dimension) throw std::runtime_error("kernel dimension too large for array");

	const std::size_t overlap_start_dim_idx = dimension - window_shape.size();
	auto& array_shape = array.shape();
	auto& array_strides = array.strides();

	shape_type new_shape(dimension + window_shape.size());
	strides_type new_strides(dimension + window_shape.size());

	// Copy the untouched batches over.
	std::copy_n(array_shape.begin(), overlap_start_dim_idx, new_shape.begin());
	std::copy_n(array_strides.begin(), overlap_start_dim_idx, new_strides.begin());
	// Copy the window shape over.
	std::copy_n(window_shape.begin(), window_shape.size(), new_shape.begin() + dimension);
	// Copy the strides over. They are the same for the old and new dimensions.
	std::copy_n(array_strides.begin(), array_strides.size(), new_strides.begin());
	std::copy_n(array_strides.begin(), window_shape.size(), new_strides.begin() + dimension);

	// Now for the overlapping parts. That's just shape for the array.
	for (std::size_t array_idx = overlap_start_dim_idx; array_idx < dimension; ++array_idx) {
		const std::size_t window_idx = array_idx - overlap_start_dim_idx;

		if (window_shape[window_idx] > array_shape[array_idx]) throw std::runtime_error("kernel axis too large for array");

		new_shape[array_idx] = array_shape[array_idx] - window_shape[array_idx] + 1;
	}

	return as_strided(array, new_shape, new_strides);
}

void va::convolve(VStoreAllocator& allocator, const VArrayTarget& target, const VArray& array, const VArray& kernel) {
	// TODO Could use support for kernel size > array, which is still a valid convolution.
	// In that case it's just the kernel that acts as the array.

	// Simple 'direct' method just involves sum_product and sliding window view.
	const std::size_t convolve_dimensions = kernel.dimension();

	const auto sliding_view = sliding_window_view(array, kernel.shape());

	axes_type axes(convolve_dimensions);
	for (int i = 0; i < convolve_dimensions; i++) axes[i] = -convolve_dimensions + i;

	va::sum_product(allocator, target, sliding_view->data, kernel.data, &axes);
}
