#include "round.hpp"

#include <utility>                                       // for move
#include "varray.hpp"                             // for VArray, VArr...
#include "vcompute.hpp"                            // for XFunction
#include "vpromote.hpp"                                    // for num_function...
#include "xtensor/xlayout.hpp"                           // for layout_type
#include "xtensor/xmath.hpp"                             // for ceil_fun
#include "xtensor/xoperation.hpp"                        // for make_xfunction

using namespace va;

// Not a NumPy ufunc because it takes a 'decimals' parameter.
void va::round(VStoreAllocator& allocator, const VArrayTarget& target, const VData& array) {
	xoperation_single<
		Feature::round,
		promote::reject_complex<promote::num_function_result_in_same_out<xt::math::round_fun>>
	>(
		va::XFunction<xt::math::round_fun> {},
		allocator,
		target,
		array
	);
}
