#ifndef NUMDOT_NDARRAY_H
#define NUMDOT_NDARRAY_H

#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"
#include "xtl/xvariant.hpp"

#ifdef WIN32
#include <windows.h>
#endif

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/input_event_key.hpp>
#include <godot_cpp/classes/tile_map.hpp>
#include <godot_cpp/classes/tile_set.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

using namespace godot;

using NDArrayVariant = xtl::variant<
	xt::xarray<double_t>,
 	xt::xarray<float_t>,
 	xt::xarray<int8_t>,
 	xt::xarray<int16_t>,
 	xt::xarray<int32_t>,
 	xt::xarray<int64_t>,
 	xt::xarray<uint8_t>,
 	xt::xarray<uint16_t>,
 	xt::xarray<uint32_t>,
 	xt::xarray<uint64_t>
>;

using NDArrayVariantContainedTypes = std::tuple<
	double_t,
 	float_t,
 	int8_t,
 	int16_t,
 	int32_t,
 	int64_t,
 	uint8_t,
 	uint16_t,
 	uint32_t,
 	uint64_t
>;

class NDArray : public Object {
	GDCLASS(NDArray, Object)

private:

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	enum DType {
		Double,
		Float,
		Int8,
		Int16,
		Int32,
		Int64,
		UInt8,
		UInt16,
		UInt32,
		UInt64,
	};

	// We need a shared pointer because things like asarray can return either a view or an array
	std::shared_ptr<NDArrayVariant> array;

	NDArray();
	NDArray(std::shared_ptr<NDArrayVariant> array) : array(array) {};
	~NDArray();

	DType dtype();
};

VARIANT_ENUM_CAST(NDArray::DType);

#endif
