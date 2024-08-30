#ifndef GDEXAMPLE_H
#define GDEXAMPLE_H

#include <godot_cpp/classes/object.hpp>

#define XTENSOR_USE_XSIMD
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtl/xvariant.hpp"

namespace godot {

using NDArrayVariant = xtl::variant<
	xt::xarray<double>,
 	xt::xarray<float>
>;

class NDArray : public Object {
	GDCLASS(NDArray, Object)

private:
	NDArrayVariant array;

protected:
	static void _bind_methods();

public:
	NDArray();
	~NDArray();
};

}

#endif
