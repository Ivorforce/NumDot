#ifndef VCARRAY_HPP
#define VCARRAY_HPP

namespace va::util {
	template<typename T, typename Compute>
	void fill_c_array_flat(T target, const Compute& carray) {
		if constexpr (!std::is_convertible_v<typename Compute::value_type, std::remove_pointer_t<std::decay_t<T>>>) {
			throw std::runtime_error("Cannot promote in this way.");
		}
		else {
			// Similar logic to va::assign with VScalar
			if (carray.layout() == xt::layout_type::row_major || carray.layout() == xt::layout_type::any)
			{
				// Contiguous assign.
				std::copy(carray.linear_begin(), carray.linear_end(), target);
			}
			else if (carray.dimension() == 1) {
				// Strided assign.
				const auto stride = carray.strides()[0];
				auto ptr = carray.linear_begin();
				const auto end = ptr + carray.shape()[0] * stride;

				for (; ptr < end; ptr += stride, ++target) {
					*target = *ptr;
				}
			}
			else
			{
				// Stepper assign.
				std::copy(carray.begin(), carray.end(), target);
			}
		}
	}

	template<typename T>
	void fill_c_array_flat(T target, const va::VData& array) {
		std::visit(
			[target](auto& carray) {
				fill_c_array_flat(target, carray);
			}, array
		);
	}

	template<typename T>
	auto adapt_c_array(T&& ptr, const va::shape_type& shape) {
		const auto size = std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
		return xt::adapt<xt::layout_type::dynamic, T, xt::no_ownership, va::shape_type>(
			std::forward<T>(ptr),
			size,
			xt::no_ownership(),
			shape,
			shape.size() < 2 ? xt::layout_type::any : xt::layout_type::row_major
		);
	}
}

#endif //VCARRAY_HPP
