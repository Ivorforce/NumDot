#include "ndrandomgenerator.hpp"

#include <gdconvert/conversion_ints.hpp>             // for variants_to_axes
#include <vatensor/linalg.hpp>                       // for reduce_dot, dot
#include <vatensor/vassign.hpp>                      // for assign
#include <algorithm>                               // for copy
#include <cstddef>                                 // for size_t
#include <stdexcept>                               // for runtime_error
#include <variant>                                 // for visit
#include <vatensor/xtensor_store.hpp>

#include "gdconvert/conversion_array.hpp"            // for fill_c_array_flat
#include "gdconvert/conversion_slice.hpp"            // for variants_to_slice_...
#include "gdconvert/conversion_string.hpp"           // for xt_to_string
#include "godot_cpp/classes/global_constants.hpp"  // for MethodFlags
#include "godot_cpp/core/class_db.hpp"             // for D_METHOD, ClassDB
#include "godot_cpp/core/error_macros.hpp"         // for ERR_FAIL_COND_V_MSG
#include "godot_cpp/core/memory.hpp"               // for _post_initialize
#include "godot_cpp/variant/string_name.hpp"       // for StringName
#include "godot_cpp/variant/variant.hpp"           // for Variant
#include "nd.hpp"                                    // for nd
#include "vatensor/varray.hpp"                       // for VArray, VArrayTarget
#include "xtensor/core/xiterator.hpp"                   // for operator==
#include "xtensor/views/xstrided_view.hpp"               // for xstrided_slice_vector
#include "xtl/xiterator_base.hpp"                  // for operator!=

using namespace godot;

void NDRandomGenerator::_bind_methods() {
	godot::ClassDB::bind_method(D_METHOD("spawn", "n"), &NDRandomGenerator::spawn);

	godot::ClassDB::bind_method(D_METHOD("random", "shape", "dtype"), &NDRandomGenerator::random, DEFVAL(PackedByteArray()), DEFVAL(va::DType::Float64));
	godot::ClassDB::bind_method(D_METHOD("integers", "low_or_high", "high", "shape", "dtype", "endpoint"), &NDRandomGenerator::integers, DEFVAL(0), DEFVAL(nullptr), DEFVAL(PackedByteArray()), DEFVAL(va::DType::Int64), DEFVAL(false));
	godot::ClassDB::bind_method(D_METHOD("randn", "shape", "dtype"), &NDRandomGenerator::randn, DEFVAL(PackedByteArray()), DEFVAL(va::DType::Float64));
}

NDRandomGenerator::NDRandomGenerator() = default;

NDRandomGenerator::~NDRandomGenerator() = default;

TypedArray<NDRandomGenerator> NDRandomGenerator::spawn(const int64_t n) {
	TypedArray<NDRandomGenerator> generators;
	generators.resize(n);
	for (int i = 0; i < n; ++i) {
		generators[i] = memnew(NDRandomGenerator(engine.spawn()));
	}
	return generators;
}

String NDRandomGenerator::_to_string() const {
	return xt_to_string(engine.engine);
}

Ref<NDArray> NDRandomGenerator::random(const Variant& shape, const va::DType dtype) {
	try {
		const auto shape_array = variant_to_shape(shape);

		return { memnew(NDArray(engine.random_floats(va::store::default_allocator, shape_array, dtype))) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> NDRandomGenerator::integers(const int64_t low_or_high, const Variant& high, const Variant& shape, const va::DType dtype, const bool endpoint) {
	try {
		const auto shape_array = variant_to_shape(shape);

		switch (high.get_type()) {
			case Variant::Type::NIL:
				return { memnew(NDArray(engine.random_integers(va::store::default_allocator, 0, low_or_high, shape_array, dtype, endpoint))) };
			case Variant::Type::INT:
				return { memnew(NDArray(engine.random_integers(va::store::default_allocator, low_or_high, static_cast<int64_t>(high), shape_array, dtype, endpoint))) };
			default:
				ERR_FAIL_V_MSG({}, "high is not an int");
		}

	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}

Ref<NDArray> NDRandomGenerator::randn(const Variant& shape, const va::DType dtype) {
	try {
		const auto shape_array = variant_to_shape(shape);

		return { memnew(NDArray(engine.random_normal(va::store::default_allocator, shape_array, dtype))) };
	}
	catch (std::runtime_error& error) {
		ERR_FAIL_V_MSG({}, error.what());
	}
}