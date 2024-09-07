#ifndef NUMDOT_NDARRAY_H
#define NUMDOT_NDARRAY_H

#include "xtensor/xarray.hpp"
#include "xtensor/xlayout.hpp"

#ifdef WIN32
#include <windows.h>
#endif

#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/binder_common.hpp>

#include "xtv.h"

using namespace godot;

class NDArray : public RefCounted {
	GDCLASS(NDArray, RefCounted)

private:

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	// We need a shared pointer because things like asarray can return either a view or an array
	std::shared_ptr<xtv::XTVariant> array;

	NDArray();
	explicit NDArray(std::shared_ptr<xtv::XTVariant> array) : array(std::move(array)) {};
	~NDArray() override;

	[[nodiscard]] xtv::DType dtype() const;
	[[nodiscard]] PackedInt64Array shape() const;
	[[nodiscard]] uint64_t size() const;
	[[nodiscard]] uint64_t array_size_in_bytes() const;
	[[nodiscard]] uint64_t ndim() const;
	
	// Subscript not available, i think. See object's set_bind / get_bind:
	// I think godot assumes that all [] accesses are keypaths.
	// https://github.com/godotengine/godot/blob/514c564a8c855d798ec6b5a52860e5bca8d57bc9/core/object/object.h#L643
	void set(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	Ref<NDArray> get(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	double_t get_float(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);
	int64_t get_int(const Variant **args, GDExtensionInt arg_count, GDExtensionCallError &error);

	[[nodiscard]] Variant as_type(xtv::DType dtype) const;

	[[nodiscard]] double_t to_float() const;
	[[nodiscard]] int64_t to_int() const;

	[[nodiscard]] PackedFloat32Array to_packed_float32_array() const;
	[[nodiscard]] PackedFloat64Array to_packed_float64_array() const;
	[[nodiscard]] PackedByteArray to_packed_byte_array() const;
	[[nodiscard]] PackedInt32Array to_packed_int32_array() const;
	[[nodiscard]] PackedInt64Array to_packed_int64_array() const;

	[[nodiscard]] Array to_godot_array() const;
};

#endif
