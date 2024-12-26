#include "vcall.hpp"

#include "array_store.hpp"
#include "create.hpp"
#include "vassign.hpp"
#include "ufunc/ufunc_features.hpp"

using namespace va;

using Dummy = char;
using UnaryDummy = void (*)(Dummy& a, const Dummy& b);
using BinaryDummy = void (*)(Dummy& a, const Dummy& b, const Dummy& c);

inline std::shared_ptr<VArray> copy_as_dtype(VStoreAllocator& allocator, const VData& a, DType dtype) {
	return va::copy_as_dtype(allocator, a, dtype);
}

inline VScalar copy_as_dtype(VStoreAllocator& allocator, const VScalar& a, DType dtype) {
	return va::static_cast_scalar(a, dtype);
}

inline const VScalar& deref(const VScalar& a) {
	return a;
}

inline const VData& deref(const std::shared_ptr<VArray>& a) {
	return a->data;
}

VData& va::evaluate_target(VStoreAllocator& allocator, const VArrayTarget& target, DType dtype, const shape_type& result_shape, std::shared_ptr<VArray>& temp) {
	if (const auto target_data = std::get_if<VData*>(&target)) {
		VData& data = **target_data;
		if (!xt::broadcastable(result_shape, va::shape(data))) {
			throw std::runtime_error("Incompatible shape of tensor destination");
		}
		if (va::dtype(data) == dtype) {
			return data;
		}

		temp = va::empty(allocator, dtype, result_shape);
		return temp->data;
	}
	else {
		auto& target_varray = *std::get<std::shared_ptr<VArray>*>(target);
		target_varray = va::empty(allocator, dtype, result_shape);
		return target_varray->data;
	}
}

void va::call_ufunc_unary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a) {
	DType a_type = va::dtype(a);

	const auto& ufunc = table[a_type];
	if (ufunc.function_ptr == nullptr) throw std::runtime_error("Unsupported dtype for ufunc.");

	std::shared_ptr<VArray> temp(nullptr);
	auto& target_ = evaluate_target(allocator, target, ufunc.output_dtype, va::shape(a), temp);

	if (a_type == ufunc.input_types[0]) {
		reinterpret_cast<UnaryDummy>(ufunc.function_ptr)(
			reinterpret_cast<Dummy&>(target_),
			reinterpret_cast<const Dummy&>(a)
		);
	}
	else {
		const auto a_ = copy_as_dtype(allocator, a, ufunc.input_types[0]);
		reinterpret_cast<UnaryDummy>(ufunc.function_ptr)(
			reinterpret_cast<Dummy&>(target_),
			reinterpret_cast<const Dummy&>(a_->data)
		);
	}

	if (temp != nullptr) {
		// We wrote to temp because the return type mismatched; now we need to resolve that.
		va::assign(*std::get<va::VData*>(target), temp->data);
	}
}

template <typename A, typename B>
void call_ufunc_binary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const shape_type& result_shape, const A& a, const B& b) {
	DType a_type = va::dtype(a);
	DType b_type = va::dtype(b);

	const auto& ufunc = table[a_type][b_type];
	if (ufunc.function_ptr == nullptr) throw std::runtime_error("Unsupported dtype for ufunc.");

	std::shared_ptr<VArray> temp(nullptr);
	auto& target_ = evaluate_target(allocator, target, ufunc.output_dtype, result_shape, temp);

	switch ((static_cast<uint8_t>(a_type == ufunc.input_types[0]) << 1) | static_cast<uint8_t>(b_type == ufunc.input_types[1])) {
		case 0b11: {
			reinterpret_cast<BinaryDummy>(ufunc.function_ptr)(
				reinterpret_cast<Dummy&>(target_),
				reinterpret_cast<const Dummy&>(a),
				reinterpret_cast<const Dummy&>(b)
			);
			break;
		}
		case 0b10: {
			const auto b_ = ::copy_as_dtype(allocator, b, ufunc.input_types[1]);
			reinterpret_cast<BinaryDummy>(ufunc.function_ptr)(
				reinterpret_cast<Dummy&>(target_),
				reinterpret_cast<const Dummy&>(a),
				reinterpret_cast<const Dummy&>(::deref(b_))
			);
			break;
		}
		case 0b01: {
			const auto a_ = ::copy_as_dtype(allocator, a, ufunc.input_types[0]);
			reinterpret_cast<BinaryDummy>(ufunc.function_ptr)(
				reinterpret_cast<Dummy&>(target_),
				reinterpret_cast<const Dummy&>(::deref(a_)),
				reinterpret_cast<const Dummy&>(b)
			);
			break;
		}
		case 0b00: {
			const auto a_ = ::copy_as_dtype(allocator, a, ufunc.input_types[0]);
			const auto b_ = ::copy_as_dtype(allocator, b, ufunc.input_types[1]);
			reinterpret_cast<BinaryDummy>(ufunc.function_ptr)(
				reinterpret_cast<Dummy&>(target_),
				reinterpret_cast<const Dummy&>(::deref(a_)),
				reinterpret_cast<const Dummy&>(::deref(b_))
			);
			break;
		}
	}

	if (temp != nullptr) {
		// We wrote to temp because the return type mismatched; now we need to resolve that.
		va::assign(*std::get<VData*>(target), temp->data);
	}
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VData& b) {
	const auto& a_shape = va::shape(a);
	const auto& b_shape = va::shape(b);

	auto result_shape = shape_type(std::max(a_shape.size(), b_shape.size()));
	std::fill_n(result_shape.begin(), result_shape.size(), std::numeric_limits<shape_type::value_type>::max());
	xt::broadcast_shape(a_shape, result_shape);
	xt::broadcast_shape(b_shape, result_shape);

	::call_ufunc_binary(allocator, table, target, result_shape, a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VScalar& a, const VData& b) {
	::call_ufunc_binary(allocator, table, target, va::shape(b), a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VScalar& b) {
	::call_ufunc_binary(allocator, table, target, va::shape(a), a, b);
}
