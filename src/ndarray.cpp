#include "ndarray.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <iostream>

// From https://github.com/xtensor-stack/xtensor/issues/1413
template <class E>
std::string xt_to_string(const xt::xexpression<E>& e)
{
    std::ostringstream out;
    out << e;
    return out.str();
}

using namespace godot;

void NDArray::_bind_methods() {
}

NDArray::NDArray() {
	array = xt::xarray<double>
	{{1.0, 2.0, 3.0},
	{2.0, 5.0, 7.0},
	{2.0, 5.0, 7.0}};

	xt::xarray<double> arr2
	{5.0, 6.0, 7.0};

	xt::xarray<double> res = xt::view(xtl::get<xt::xarray<double>>(array), 1) + arr2;

	godot::UtilityFunctions::print(String(xt_to_string(res).c_str()));
}

NDArray::~NDArray() {
	// Add your cleanup here.
}
