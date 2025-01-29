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
	_call_vfunc_binary(allocator, table, target, result_shape, a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VData& b) {
	call_vfunc_binary(allocator, table, target, a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VScalar& a, const VData& b) {
	_call_vfunc_binary(allocator, table, target, va::shape(b), a, b);
}

void va::call_ufunc_binary(VStoreAllocator& allocator, const vfunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VScalar& b) {
	_call_vfunc_binary(allocator, table, target, va::shape(a), a, b);
}
