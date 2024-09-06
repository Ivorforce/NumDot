#ifndef NUMDOT_NDRANGE_H
#define NUMDOT_NDRANGE_H

#ifdef WIN32
#include <windows.h>
#endif

#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/binder_common.hpp>

#include "xtensor/xslice.hpp"

using namespace godot;

using range_part = std::variant<xt::placeholders::xtuph, std::ptrdiff_t>;

class NDRange : public RefCounted {
	GDCLASS(NDRange, RefCounted)

private:

protected:
	static void _bind_methods();
	String _to_string() const;

public:
	// We need a shared pointer because things like asarray can return either a view or an array
	range_part start = xt::placeholders::xtuph{};
	range_part stop = xt::placeholders::xtuph{};
	range_part step = xt::placeholders::xtuph{};

	NDRange(range_part start, range_part stop, range_part step) : start(start), stop(stop), step(step) {};

	NDRange();
	~NDRange();
};

#endif
