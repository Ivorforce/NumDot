#ifndef NUMDOT_ND_H
#define NUMDOT_ND_H

#include <godot_cpp/classes/object.hpp>

#include "ndarray.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtl/xvariant.hpp"

namespace godot {

// From https://github.com/xtensor-stack/xtensor/issues/1413
template <class E>
String xt_to_string(const xt::xexpression<E>& e)
{
    std::ostringstream out;
    out << e;
    return String(out.str().c_str());
}

class ND : public Object {
	GDCLASS(ND, Object)

protected:
	static void _bind_methods();

public:
	ND();
	~ND();

	static bool _asshape(Variant shape, std::shared_ptr<xt::xarray<uint64_t>> &target);
	static bool _asarray(Variant array, std::shared_ptr<NDArrayVariant> &target);

	static Variant asarray(Variant array);
	static Variant zeros(Variant shape);
	static Variant ones(Variant shape);

	static Variant add(Variant a, Variant b);
};

}

#endif
