#include "ndarray.h"

#include <godot_cpp/godot.hpp>
#include "xtensor/xtensor.hpp"

#include "xtv.h"
#include "nd.h"
#include "conversion_array.h"
#include "conversion_slice.h"

using namespace godot;

void NDRange::_bind_methods() {
}

std::string arg_to_string(std::ptrdiff_t v) { return std::to_string(v); };
std::string arg_to_string(xt::placeholders::xtuph v) { return "null"; };

String NDRange::_to_string() const {
	return std::visit([](auto a, auto b, auto c){
		using A = std::decay_t<decltype(a)>;
		using B = std::decay_t<decltype(b)>;
		using C = std::decay_t<decltype(c)>;
		
		std::stringstream ss;
		ss << "nd.range(" << arg_to_string(a) << ", " << arg_to_string(b) << ", " << arg_to_string(c) << ")";

		return String(ss.str().c_str());
	}, start, stop, step);
}

NDRange::NDRange() {
}

NDRange::~NDRange() {
}
