#ifndef VATENSOR_UFUNC_HPP
#define VATENSOR_UFUNC_HPP

#include "vatensor/dtype.hpp"

namespace va::vfunc {
	template <int N>
	struct UFunc {
		std::array<va::DType, N> input_types;
		va::DType output_dtype;
		void* function_ptr;
	};
}

namespace va::vfunc::tables {
	using UFuncTableUnary = std::array<UFunc<1>, DTypeMax>;
	using UFuncTableBinary = std::array<std::array<UFunc<2>, DTypeMax>, DTypeMax>;
}

#endif //VATENSOR_UFUNC_HPP
