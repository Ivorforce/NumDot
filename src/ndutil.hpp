#ifndef NDUTIL_HPP
#define NDUTIL_HPP

#include "godot_cpp/variant/string_name.hpp"  // for StringName
#include "godot_cpp/variant/variant.hpp"  // for Variant
#include "godot_cpp/core/class_db.hpp"  // for Variant
#include "godot_cpp/core/object.hpp"  // for Variant

using namespace godot;

StringName axis_all();
StringName newaxis();
StringName ellipsis();
StringName no_value();

bool is_no_value(const Variant& variant);

namespace numdot {
	struct VarargMethodDefinition {
		StringName name;
		std::vector<PropertyInfo> args;
		VarargMethodDefinition() = default;
		explicit VarargMethodDefinition(StringName&& p_name) : name(std::forward<StringName>(p_name)) {}
	};

	template<typename... Args>
	VarargMethodDefinition VD_METHOD(StringName&& p_name, Args... args) {
		VarargMethodDefinition md(std::forward<StringName>(p_name));
		(md.args.push_back(args), ...);
		return md;
	}

	template <typename M, typename... DefaultValues>
	MethodBind *bind_vararg_method(VarargMethodDefinition d_method, M p_method, bool p_return_nil_is_variant = true, DefaultValues... defaults) {
		MethodInfo mi;
		mi.name = d_method.name;
		mi.arguments = d_method.args;
		mi.flags = METHOD_FLAG_VARARG;

		// TODO mi.flags and first param of bind_vararg_method is ignored.
		return ClassDB::bind_vararg_method(
			mi.flags,  // This is ignored, lol
			d_method.name,
			p_method,
			mi,
			std::vector<Variant>(defaults...),
			p_return_nil_is_variant
		);
	}
}

#endif //NDUTIL_HPP
