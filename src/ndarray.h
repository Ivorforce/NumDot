#ifndef NUMDOT_NDARRAY_H
#define NUMDOT_NDARRAY_H

#include <godot_cpp/classes/object.hpp>

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

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	NDArray();
	NDArray(std::shared_ptr<NDArrayVariant> array) : array(array) {};
	~NDArray();
	
	// We need a shared pointer because things like asarray can return either a view or an array
	std::shared_ptr<NDArrayVariant> array;
};

}

#endif
