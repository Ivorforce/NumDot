#include "vcall.hpp"

#include "array_store.hpp"
#include "create.hpp"
#include "vfunc/ufunc_features.hpp"

using namespace va;

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

void va::call_ufunc_unary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a) {
	call_vfunc_unary(allocator, table, target, a);
}

template <typename A, typename B>
void call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const shape_type& result_shape, const A& a, const B& b) {
	call_vfunc_binary(allocator, table, target, result_shape, a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VData& b) {
	const auto& a_shape = va::shape(a);
	const auto& b_shape = va::shape(b);

	auto result_shape = shape_type(std::max(a_shape.size(), b_shape.size()));
	std::fill_n(result_shape.begin(), result_shape.size(), std::numeric_limits<shape_type::value_type>::max());
	xt::broadcast_shape(a_shape, result_shape);
	xt::broadcast_shape(b_shape, result_shape);

	::call_ufunc_binary(allocator, table, target, result_shape, a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VScalar& a, const VData& b) {
	::call_ufunc_binary(allocator, table, target, va::shape(b), a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VScalar& b) {
	::call_ufunc_binary(allocator, table, target, va::shape(a), a, b);
}
