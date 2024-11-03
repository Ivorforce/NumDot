#include "linalg.hpp"

#include <cstddef>                // for ptrdiff_t
#include <optional>               // for optional
#include <stdexcept>              // for runtime_error
#include <vector>                 // for vector
#include "reduce.hpp"               // for sum
#include "varray.hpp"      // for VArray, VArrayTarget, VScalar, axes...
#include "vmath.hpp"                // for multiply
#include "xtensor/xslice.hpp"     // for all, ellipsis, newaxis, xall_tag

void va::dot(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
	if (a.dimension() == 0 || b.dimension() == 0) {
		va::multiply(allocator, target, a, b);
		return;
	}
	if (a.dimension() <= 2 && b.dimension() <= 2) {
		va::matmul(allocator, target, a, b);
		return;
	}

	throw std::runtime_error("tensordot is not yet implemented");
}

void va::matmul(VStoreAllocator& allocator, VArrayTarget target, const VArray& a, const VArray& b) {
	if (a.dimension() == 0 || b.dimension() == 0) {
		throw std::runtime_error("matmul does not accept scalars");
	}
	if (b.dimension() == 1) {
		va::reduce_dot(allocator, target, a, b, {-1});
		return;
	}
	if (a.dimension() == 1) {
		const auto promoted_a = a.sliced({xt::all(), xt::newaxis()});
		va::reduce_dot(allocator, target, *promoted_a, b, {-2});
		return;
	}

	const std::shared_ptr<VArray> a_broadcast = a.sliced({ xt::ellipsis(), xt::newaxis() });
	const std::shared_ptr<VArray> b_broadcast = b.sliced({ xt::ellipsis(), xt::newaxis(), xt::all(), xt::all() });

	reduce_dot(allocator, target, *a_broadcast, *b_broadcast, std::vector<std::ptrdiff_t> { -2 });
}
