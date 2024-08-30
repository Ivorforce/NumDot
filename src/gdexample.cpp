#include "gdexample.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <iostream>

#define XTENSOR_USE_XSIMD
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

// From https://github.com/xtensor-stack/xtensor/issues/1413
template <class E>
std::string xt_to_string(const xt::xexpression<E>& e)
{
    std::ostringstream out;
    out << e;
    return out.str();
}

using namespace godot;

void GDExample::_bind_methods() {
}

GDExample::GDExample() {
	// Initialize any variables here.
	time_passed = 0.0;

	xt::xarray<double> arr1
	{{1.0, 2.0, 3.0},
	{2.0, 5.0, 7.0},
	{2.0, 5.0, 7.0}};

	xt::xarray<double> arr2
	{5.0, 6.0, 7.0};

	xt::xarray<double> res = xt::view(arr1, 1) + arr2;

	godot::UtilityFunctions::print(String(xt_to_string(res).c_str()));
}

GDExample::~GDExample() {
	// Add your cleanup here.
}
