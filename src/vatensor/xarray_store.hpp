#ifndef VSTORE_HPP
#define VSTORE_HPP

#include <memory>
#include <cmath>                           // for double_t, float_t
#include <complex>
#include <cstddef>                         // for size_t
#include <cstdint>                         // for int16_t, int32_t, int64_t

#include "varray.hpp"

namespace va::store {
	using XArrayStoreVariant = std::variant<
		array_case<bool>,
		array_case<float_t>,
		array_case<double_t>,
		array_case<std::complex<float_t>>,
		array_case<std::complex<double_t>>,
		array_case<int8_t>,
		array_case<int16_t>,
		array_case<int32_t>,
		array_case<int64_t>,
		array_case<uint8_t>,
		array_case<uint16_t>,
		array_case<uint32_t>,
		array_case<uint64_t>
	>;

	class XArrayStore : public VStore {
		public:
			XArrayStoreVariant array;
			explicit XArrayStore(XArrayStoreVariant&& array) : array(std::forward<XArrayStoreVariant>(array)) {}
	};

	// For deducted V, from xexpressions
	template<typename T, typename V = typename std::decay_t<T>::value_type>
	static array_case<V> make_store(T&& data) {
		return array_case<V>(std::forward<T>(data));
	}

	template<typename V>
	static array_case<V> make_store(std::initializer_list<V> data) {
		return array_case<V>(data);
	}

	template<typename V>
	static std::shared_ptr<VArray> from_store(array_case<V>&& array) {
		const auto data_offset = static_cast<std::ptrdiff_t>(array.data_offset());

		auto compute = make_compute<V*>(
			array.data() + array.data_offset(),  // Offset should be 0, but you know...
			array.shape(),
			array.strides(),
			array.layout()
		);

		return std::make_shared<VArray>(
			VArray {
				std::shared_ptr<VStore>(std::make_shared<XArrayStore>(XArrayStore { XArrayStoreVariant { std::forward<array_case<V>>(array) } })),
				compute,
				data_offset
			}
		);
	}
}

#endif //VSTORE_HPP
