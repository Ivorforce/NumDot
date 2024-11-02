#include "packed_array_store.hpp"

template struct numdot::PackedArrayStore<godot::PackedByteArray>;
template struct numdot::PackedArrayStore<godot::PackedColorArray>;
template struct numdot::PackedArrayStore<godot::PackedFloat32Array>;
template struct numdot::PackedArrayStore<godot::PackedFloat64Array>;
template struct numdot::PackedArrayStore<godot::PackedInt32Array>;
template struct numdot::PackedArrayStore<godot::PackedInt64Array>;
template struct numdot::PackedArrayStore<godot::PackedVector2Array>;
template struct numdot::PackedArrayStore<godot::PackedVector3Array>;
template struct numdot::PackedArrayStore<godot::PackedVector4Array>;
