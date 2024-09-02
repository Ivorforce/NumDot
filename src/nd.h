#ifndef NUMDOT_ND_H
#define NUMDOT_ND_H

#include <godot_cpp/godot.hpp>

#include "ndarray.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "xtv.h"

using namespace godot;

// From https://github.com/xtensor-stack/xtensor/issues/1413
template <class E>
String xt_to_string(const xt::xexpression<E>& e)
{
    std::ostringstream out;
    out << e;
    return String(out.str().c_str());
}

class nd : public Object {
	GDCLASS(nd, Object)

protected:
	static void _bind_methods();

public:
	nd();
	~nd();

	static NDArray::DType dtype(Variant array);
	static PackedInt64Array shape(Variant array);
	static uint64_t size(Variant array);
	static uint64_t ndim(Variant array);
	
	static Variant as_type(Variant array, NDArray::DType dtype);

	static Variant as_array(Variant array, NDArray::DType dtype = NDArray::DType::DTypeMax);
	static Variant array(Variant array, NDArray::DType dtype = NDArray::DType::DTypeMax);
	static Variant zeros(Variant shape, NDArray::DType dtype = NDArray::DType::Float64);
	static Variant ones(Variant shape, NDArray::DType dtype = NDArray::DType::Float64);

	static Variant add(Variant a, Variant b);
	static Variant subtract(Variant a, Variant b);
	static Variant multiply(Variant a, Variant b);
	static Variant divide(Variant a, Variant b);
};

#endif
