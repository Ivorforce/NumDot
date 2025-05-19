#ifndef PACKED_ARRAY_STORE_HPP
#define PACKED_ARRAY_STORE_HPP

#include "godot_cpp/variant/packed_byte_array.hpp"     // for PackedByteArray
#include "godot_cpp/variant/packed_color_array.hpp"    // for PackedColorArray
#include "godot_cpp/variant/packed_float32_array.hpp"  // for PackedFloat32A...
#include "godot_cpp/variant/packed_float64_array.hpp"  // for PackedFloat64A...
#include "godot_cpp/variant/packed_int32_array.hpp"    // for PackedInt32Array
#include "godot_cpp/variant/packed_int64_array.hpp"    // for PackedInt64Array
#include "godot_cpp/variant/packed_vector2_array.hpp"  // for PackedVector2A...
#include "godot_cpp/variant/packed_vector3_array.hpp"  // for PackedVector3A...
#include "godot_cpp/variant/packed_vector4_array.hpp"  // for PackedVector4A...
#include <vatensor/varray.hpp>

namespace numdot {
	template<class T>
	auto get_packed_content_type(const T& packed) {
		using C = std::decay_t<decltype(*packed.ptr())>;
		if constexpr (std::is_same_v<C, godot::Vector2>) return real_t{};
		else if constexpr (std::is_same_v<C, godot::Vector3>) return real_t{};
		else if constexpr (std::is_same_v<C, godot::Vector4>) return real_t{};
		else if constexpr (std::is_same_v<C, godot::Color>) return float_t{};
		else return C{};
	}

	template <typename Array>
	class PackedArrayStore : public va::VStore {
	public:
		// Keep in mind this is copy-on-write.
		Array array;
		explicit PackedArrayStore(Array&& array) : array(std::forward<Array>(array)) {}

		void* data() override { return const_cast<void*>(static_cast<const void*>(array.ptr())); }
		va::DType dtype() override { return va::dtype_of_type<decltype(get_packed_content_type(array))>(); }
		std::size_t size() override { return static_cast<std::size_t>(array.size()); }
		void prepare_write(va::VData& data, std::ptrdiff_t data_offset) override;
	};

	template<typename Array>
	void PackedArrayStore<Array>::prepare_write(va::VData& data, std::ptrdiff_t data_offset) {
		auto* ptrw = array.ptrw();  // May create copy, so we need to update the data pointer.
		std::visit([ptrw, data_offset](auto& carray) {
			using V = typename std::decay_t<decltype(carray)>::value_type;
			carray.reset_buffer(reinterpret_cast<V*>(ptrw) + data_offset, carray.storage().size());
		}, data);
	}

	template<typename Array, typename C>
	std::shared_ptr<va::VArray> varray_from_packed(C&& compute, Array&& array) {
		// A bit fishy to initialize the compute beforehand, but it's guaranteed to point to the same data so far because it's COW.
		auto store = std::make_shared<PackedArrayStore<Array>>(PackedArrayStore<Array> { std::forward<Array>(array) });

		return std::make_shared<va::VArray>(
			va::VArray {
				std::shared_ptr<va::VStore>(store),
				std::forward<C>(compute),
				0
			}
		);
	}

	using VStorePackedByteArray = PackedArrayStore<godot::PackedByteArray>;
	using VStorePackedColorArray = PackedArrayStore<godot::PackedColorArray>;
	using VStorePackedFloat32Array = PackedArrayStore<godot::PackedFloat32Array>;
	using VStorePackedFloat64Array = PackedArrayStore<godot::PackedFloat64Array>;
	using VStorePackedInt32Array = PackedArrayStore<godot::PackedInt32Array>;
	using VStorePackedInt64Array = PackedArrayStore<godot::PackedInt64Array>;
	using VStorePackedVector2Array = PackedArrayStore<godot::PackedVector2Array>;
	using VStorePackedVector3Array = PackedArrayStore<godot::PackedVector3Array>;
	using VStorePackedVector4Array = PackedArrayStore<godot::PackedVector4Array>;
}

#endif //PACKED_ARRAY_STORE_HPP
