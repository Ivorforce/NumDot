#include "ndrange.h"

#include <sstream>          // for operator<<, basic_stringstream, basic_ost...
#include <string>           // for char_traits, allocator, basic_string, string

using namespace godot;

void NDRange::_bind_methods() {
}

std::string arg_to_string(std::ptrdiff_t v) { return std::to_string(v); };
std::string arg_to_string(xt::placeholders::xtuph v) { return "null"; };

String NDRange::_to_string() const {
	return std::visit([](auto a, auto b, auto c){
		std::stringstream ss;
		ss << "nd.range(" << arg_to_string(a) << ", " << arg_to_string(b) << ", " << arg_to_string(c) << ")";

		return String(ss.str().c_str());
	}, start, stop, step);
}

NDRange::NDRange() {
}

NDRange::~NDRange() {
}
