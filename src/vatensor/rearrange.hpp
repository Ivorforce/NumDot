#ifndef VA_H
#define VA_H

#include <cstddef>      // for ptrdiff_t, size_t
#include <type_traits>  // for decay_t
#include <variant>      // for visit
#include "varray.hpp"     // for VArray, strides_type, axes_type, from_surrogate

namespace va {
	template<typename Visitor>
	std::shared_ptr<VArray> map(const Visitor& visitor, const VArray& varray) {
		return std::visit(
			[&varray, &visitor](const auto& read) -> std::shared_ptr<VArray> {
				using VTRead = typename std::decay_t<decltype(read)>::value_type;

				return va::from_surrogate(
					varray,
					visitor(read),
					const_cast<VTRead*>(read.data())
				);
			}, varray.data
		);
	}

	template<typename Visitor>
	VData map_compute(const Visitor& visitor, const VData& data) {
		return std::visit(
			[&visitor](const auto& read) -> VData {
				using VTRead = typename std::decay_t<decltype(read)>::value_type;

				return va::compute_from_surrogate(
					visitor(read),
					const_cast<VTRead*>(read.data())
				);
			}, data
		);
	}

	std::shared_ptr<VArray> transpose(const VArray& varray, strides_type permutation);
	std::shared_ptr<VArray> transpose(const VArray& varray);
	std::shared_ptr<VArray> swapaxes(const VArray& varray, std::ptrdiff_t a, std::ptrdiff_t b);
	std::shared_ptr<VArray> moveaxis(const VArray& varray, std::ptrdiff_t src, std::ptrdiff_t dst);
	VData moveaxis(const VData& data, std::ptrdiff_t src, std::ptrdiff_t dst);
	std::shared_ptr<VArray> flip(const VArray& varray, std::size_t axis);
	std::shared_ptr<VArray> diagonal(const VArray& varray, std::ptrdiff_t offset, std::ptrdiff_t axis1, std::ptrdiff_t axis2);
	std::shared_ptr<VArray> join_axes_into_last_dimension(const VData& varray, axes_type axes);

	std::shared_ptr<VArray> real(const std::shared_ptr<VArray>& varray);
	std::shared_ptr<VArray> imag(const std::shared_ptr<VArray>& varray);
	std::shared_ptr<VArray> complex_as_vector(const std::shared_ptr<VArray>& varray);
	std::shared_ptr<VArray> vector_as_complex(VStoreAllocator& allocator, const VArray& varray, DType dtype, bool keepdims);

	std::shared_ptr<VArray> squeeze(const std::shared_ptr<VArray>& vdata);
}

#endif //XV_H
