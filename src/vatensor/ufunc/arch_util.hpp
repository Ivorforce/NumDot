#ifndef VATENSOR_ARCH_UTIL_HPP
#define VATENSOR_ARCH_UTIL_HPP

#define DECLARE_NATIVE_UNARY(UFUNC_NAME, IN_TYPE, RETURN_TYPE)\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN_TYPE*>>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN_TYPE*>& a)

#define ADD_NATIVE_UNARY(UFUNC_NAME, IN_TYPE, RETURN_TYPE)\
tables::UFUNC_NAME[va::dtype_of_type<IN_TYPE>()] = UFunc {\
	va::dtype_of_type<IN_TYPE>(),\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN_TYPE*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
}

#define ADD_CAST_UNARY(UFUNC_NAME, MODEL_TYPE, IN0)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()] = tables::UFUNC_NAME[va::dtype_of_type<MODEL_TYPE>()]


#define DECLARE_NATIVE_BINARY(UFUNC_NAME, IN_TYPE, RETURN_TYPE)\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN_TYPE*>, compute_case<IN_TYPE*>>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN_TYPE*>& a, const compute_case<IN_TYPE*>& b);\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, IN_TYPE, compute_case<IN_TYPE*>>(compute_case<RETURN_TYPE*>& ret, const IN_TYPE& a, const compute_case<IN_TYPE*>& b);\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN_TYPE*>, IN_TYPE>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN_TYPE*>& a, const IN_TYPE& b)\

#define DECLARE_NATIVE_BINARY_COMMUTATIVE(UFUNC_NAME, IN_TYPE, RETURN_TYPE)\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN_TYPE*>, compute_case<IN_TYPE*>>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN_TYPE*>& a, const compute_case<IN_TYPE*>& b);\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN_TYPE*>, IN_TYPE>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN_TYPE*>& a, const IN_TYPE& b)\

#define ADD_NATIVE_BINARY(UFUNC_NAME, IN_TYPE, RETURN_TYPE)\
tables::UFUNC_NAME[va::dtype_of_type<IN_TYPE>()][va::dtype_of_type<IN_TYPE>()] = UFunc {\
	va::dtype_of_type<IN_TYPE>(),\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN_TYPE*>&, const compute_case<IN_TYPE*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
};\
tables::UFUNC_NAME##_scalarLeft[va::dtype_of_type<IN_TYPE>()][va::dtype_of_type<IN_TYPE>()] = UFunc {\
	va::dtype_of_type<IN_TYPE>(),\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const IN_TYPE&, const compute_case<IN_TYPE*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
};\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN_TYPE>()][va::dtype_of_type<IN_TYPE>()] = UFunc {\
	va::dtype_of_type<IN_TYPE>(),\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN_TYPE*>&, const IN_TYPE&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
}

#define ADD_NATIVE_BINARY_COMMUTATIVE(UFUNC_NAME, IN_TYPE, RETURN_TYPE)\
tables::UFUNC_NAME[va::dtype_of_type<IN_TYPE>()][va::dtype_of_type<IN_TYPE>()] = UFunc {\
	va::dtype_of_type<IN_TYPE>(),\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN_TYPE*>&, const compute_case<IN_TYPE*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
};\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN_TYPE>()][va::dtype_of_type<IN_TYPE>()] = UFunc {\
	va::dtype_of_type<IN_TYPE>(),\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN_TYPE*>&, const IN_TYPE&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
}

#define ADD_CAST_BINARY(UFUNC_NAME, MODEL_TYPE, IN0, IN1)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME[va::dtype_of_type<MODEL_TYPE>()][va::dtype_of_type<MODEL_TYPE>()];\
tables::UFUNC_NAME##_scalarLeft[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME##_scalarLeft[va::dtype_of_type<MODEL_TYPE>()][va::dtype_of_type<MODEL_TYPE>()];\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<MODEL_TYPE>()][va::dtype_of_type<MODEL_TYPE>()]

#define ADD_CAST_BINARY_COMMUTATIVE(UFUNC_NAME, MODEL_TYPE, IN0, IN1)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME[va::dtype_of_type<MODEL_TYPE>()][va::dtype_of_type<MODEL_TYPE>()];\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<MODEL_TYPE>()][va::dtype_of_type<MODEL_TYPE>()]

#endif //VATENSOR_ARCH_UTIL_HPP
