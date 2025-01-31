#include "linalg.hpp"

#include <cstddef>                // for ptrdiff_t
#include <optional>               // for optional
#include <stdexcept>              // for runtime_error
#include <vector>                 // for vector
#include <xtensor/xview.hpp>
#include "vfunc/entrypoints.hpp"
#include "create.hpp"
#include "rearrange.hpp"
#include "reduce.hpp"               // for sum
#include "util.hpp"
#include "varray.hpp"      // for VArray, VArrayTarget, VScalar, axes...
#include "vcompute.hpp"
#include "vmath.hpp"                // for multiply
#include "xtensor_store.hpp"
#include "xtensor/xslice.hpp"     // for all, ellipsis, newaxis, xall_tag


void va::dot(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
	if (va::dimension(a) == 0 || va::dimension(b) == 0) {
		va::multiply(allocator, target, a, b);
		return;
	}
	if (va::dimension(a) <= 2 && va::dimension(b) <= 2) {
		va::matmul(allocator, target, a, b);
		return;
	}

	throw std::runtime_error("tensordot is not yet implemented");
}

void assign_cross(va::VData& target, const va::VData& a, const va::VData& b, const std::ptrdiff_t target_axis) {
	const auto a_size = va::shape(a).back();
	const auto b_size = va::shape(b).back();

	auto dummy_allocator = va::store::default_allocator;  // Not needed

	const auto target_axis_normal = va::util::normalize_axis(target_axis, va::dimension(target));

	xt::xstrided_slice_vector slice(target_axis_normal + 1);
	std::fill_n(slice.begin(), target_axis_normal, xt::all());

	if (a_size == 3 && b_size == 3) {
		slice.back() = 0;
		auto t0 = va::sliced_data(target, slice);
		a0xb1_minus_a1xb0(dummy_allocator, &t0, a, b, 1, 2);

		slice.back() = 1;
		auto t1 = va::sliced_data(target, slice);
		a0xb1_minus_a1xb0(dummy_allocator, &t1, a, b, 2, 0);
	}
	else if (a_size == 2 && b_size == 3) {
		const auto b2 = va::sliced_data(b, xt::xstrided_slice_vector { xt::ellipsis(), 2 });

		{
			slice.back() = 0;
			auto t0 = va::sliced_data(target, slice);
			va::multiply(
				dummy_allocator,
				&t0,
				va::sliced_data(a, xt::xstrided_slice_vector { xt::ellipsis(), 1 }),
				b2
			);
		}

		{
			slice.back() = 1;
			auto t1 = va::sliced_data(target, slice);
			va::multiply(
				dummy_allocator,
				&t1,
				va::sliced_data(a, xt::xstrided_slice_vector { xt::ellipsis(), 0 }),
				b2
			);
			va::negative(dummy_allocator, &t1, t1);
		}
	}
	else if (a_size == 3 && b_size == 2) {
		const auto a2 = va::sliced_data(a, xt::xstrided_slice_vector { xt::ellipsis(), 2 });

		{
			slice.back() = 0;
			auto t0 = va::sliced_data(target, slice);
			va::multiply(
				dummy_allocator,
				&t0,
				a2,
				va::sliced_data(b, xt::xstrided_slice_vector { xt::ellipsis(), 1 })
			);
			va::negative(dummy_allocator, &t0, t0);
		}

		{
			slice.back() = 1;
			auto t1 = va::sliced_data(target, slice);
			va::multiply(
				dummy_allocator,
				&t1,
				a2,
				va::sliced_data(b, xt::xstrided_slice_vector { xt::ellipsis(), 0 })
			);
		}
	}
	else if (a_size != 2 || b_size != 2) {
		throw std::runtime_error("cross must be called with 2-D or 3-D axes");
	}

	slice.back() = 2;
	auto t2 = va::sliced_data(target, slice);
	a0xb1_minus_a1xb0(dummy_allocator, &t2, a, b, 0, 1);
}

void va::cross(::va::VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b, std::ptrdiff_t axisa, std::ptrdiff_t axisb, std::ptrdiff_t axisc) {
	const auto a_ = va::moveaxis(a, axisa, -1);
	const auto b_ = va::moveaxis(b, axisb, -1);

	if (va::shape(a_).back() == 2 && va::shape(b_).back() == 2) {
		// z-only
		a0xb1_minus_a1xb0(va::store::default_allocator, target, a_, b_, 0, 1);
		return;
	}

	std::visit([&allocator, &a_, &b_, axisc](auto& target) {
		using PtrType = std::decay_t<decltype(target)>;

		if constexpr (std::is_same_v<PtrType, VData*>) {
			assign_cross(*target, a_, b_, axisc);
		}
		else {
			shape_type a_shape = shape_type(va::dimension(a_) - 1);
			std::copy_n(va::shape(a_).begin(), a_shape.size(), a_shape.begin());

			shape_type b_shape = shape_type(va::dimension(b_) - 1);
			std::copy_n(va::shape(b_).begin(), b_shape.size(), b_shape.begin());

			auto result_shape = shape_type(std::max(a_shape.size(), b_shape.size()));
			std::fill_n(result_shape.begin(), result_shape.size(), std::numeric_limits<shape_type::value_type>::max());
			xt::broadcast_shape(a_shape, result_shape);
			xt::broadcast_shape(b_shape, result_shape);

			// + 1 because we the axis references the array after axis insertion
			const auto axisc_normal = va::util::normalize_axis(axisc, result_shape.size() + 1);
			result_shape.insert(result_shape.begin() + axisc_normal, 3);

			const auto result_dtype = va::dtype_common_type(va::dtype(a_), va::dtype(b_));
			*target = va::empty(allocator, result_dtype, result_shape);
			assign_cross((*target)->data, a_, b_, axisc);
		}
	}, target);
}

void va::matmul(VStoreAllocator& allocator, const VArrayTarget& target, const VData& a, const VData& b) {
	if (va::dimension(a) == 0 || va::dimension(b) == 0) {
		throw std::runtime_error("matmul does not accept scalars");
	}
	if (va::dimension(b) == 1) {
		auto axes = axes_type {-1};
		va::reduce_dot(allocator, target, a, b, &axes);
		return;
	}
	if (va::dimension(a) == 1) {
		const auto promoted_a = va::sliced_data(a, {xt::all(), xt::newaxis()});
		auto axes = axes_type {-2};
		va::reduce_dot(allocator, target, promoted_a, b, &axes);
		return;
	}

	auto a_broadcast = va::sliced_data(a, { xt::ellipsis(), xt::newaxis() });
	auto b_broadcast = va::sliced_data(b, { xt::ellipsis(), xt::newaxis(), xt::all(), xt::all() });

	auto axes = axes_type {-2};
	reduce_dot(allocator, target, a_broadcast, b_broadcast, &axes);
}
