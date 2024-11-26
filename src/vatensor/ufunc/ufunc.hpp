#ifndef VATENSOR_UFUNC_HPP
#define VATENSOR_UFUNC_HPP

#include "vatensor/dtype.hpp"

namespace va::ufunc {
	struct UFunc {
		va::DType input_type;
		va::DType output_dtype;
		void* function_ptr;
	};
}

namespace va::ufunc::tables {
	using UFuncTableUnary = std::array<UFunc, DTypeMax>;
	using UFuncTableBinary = std::array<std::array<UFunc, DTypeMax>, DTypeMax>;
}

#endif //VATENSOR_UFUNC_HPP
