class_name TestUtil
extends Node

static func to_packed(array: NDArray):
	var dtype := array.dtype()
	if dtype == nd.Int8:
		return array.to_packed_int32_array()
	if dtype == nd.Int16:
		return array.to_packed_int32_array()
	if dtype == nd.Int32:
		return array.to_packed_int64_array()
	if dtype == nd.Int64:
		return array.to_packed_int64_array()
	if dtype == nd.UInt8:
		return array.to_packed_byte_array()
	if dtype == nd.UInt16:
		return array.to_packed_int32_array()
	if dtype == nd.UInt32:
		return array.to_packed_int32_array()
	if dtype == nd.UInt64:
		return array.to_packed_int64_array()
	if dtype == nd.Float32:
		return array.to_packed_float32_array()
	if dtype == nd.Float64:
		return array.to_packed_float64_array()
	if dtype == nd.Bool:
		return array.to_packed_byte_array()
	if dtype == nd.Complex64:
		return nd.complex_as_vector(array).to_packed_vector2_array()
	if dtype == nd.Complex128:  # Not technically correct because it's not double, but eh...
		return nd.complex_as_vector(array).to_packed_vector2_array()
	assert(false)
