#ifndef VCALL_HPP
#define VCALL_HPP

#include "varray.hpp"
#include "ufunc/ufunc.hpp"

namespace va {
	VData& evaluate_target(VStoreAllocator& allocator, const VArrayTarget& target, DType dtype, const shape_type& result_shape, std::shared_ptr<VArray>& temp);

	void call_ufunc_unary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableUnary& table, const VArrayTarget& target, const VData& a);
	void call_ufunc_binary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VData& b);

	void call_ufunc_binary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VScalar& a, const VData& b);
	void call_ufunc_binary(VStoreAllocator& allocator, const ufunc::tables::UFuncTableBinary& table, const VArrayTarget& target, const VData& a, const VScalar& b);
}

#endif //VCALL_HPP
