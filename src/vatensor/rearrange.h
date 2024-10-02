#ifndef VA_H
#define VA_H

#include "auto_defines.h"

#include <cstddef>      // for ptrdiff_t, size_t
#include <type_traits>  // for decay_t
#include <variant>      // for visit
#include "varray.h"     // for VArray, strides_type, axes_type, from_surrogate

namespace va {
	template<typename Visitor>
	static std::shared_ptr<VArray> map(const Visitor& visitor, const VArray& varray) {
		return std::visit(
			[visitor](auto& store, auto& read) -> std::shared_ptr<VArray> {
				using VTStore = typename std::decay_t<decltype(*store)>::value_type;
				using VTRead = typename std::decay_t<decltype(read)>::value_type;

				if constexpr (!std::is_same_v<VTStore, VTRead>) {
					throw std::runtime_error("unexpected data type discrepancy between store and read");
				}

				return from_surrogate(store, visitor(read));
			}, varray.store, varray.read
		);
	}

	std::shared_ptr<VArray> transpose(const VArray& varray, strides_type permutation);
	std::shared_ptr<VArray> reshape(const VArray& varray, strides_type new_shape);
	std::shared_ptr<VArray> swapaxes(const VArray& varray, std::ptrdiff_t a, std::ptrdiff_t b);
	std::shared_ptr<VArray> moveaxis(const VArray& varray, std::ptrdiff_t src, std::ptrdiff_t dst);
	std::shared_ptr<VArray> flip(const VArray& varray, std::size_t axis);
	std::shared_ptr<VArray> join_axes_into_last_dimension(const VArray& varray, axes_type axes);
}

#endif //XV_H
