#ifndef VATENSOR_ARCH_UTIL_HPP
#define VATENSOR_ARCH_UTIL_HPP

#define DECLARE_NATIVE_UNARY(UFUNC_NAME, RETURN_TYPE, IN0)\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN0*>>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a)

#define ADD_NATIVE_UNARY(UFUNC_NAME, RETURN_TYPE, IN0)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()] = UFunc<1> {\
	{ va::dtype_of_type<IN0>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
}

#define ADD_CAST_UNARY(UFUNC_NAME, MODEL_IN0, IN0)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()] = tables::UFUNC_NAME[va::dtype_of_type<MODEL_IN0>()]


#define DECLARE_NATIVE_BINARY(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN0*>, compute_case<IN1*>>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const compute_case<IN1*>& b);\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, IN0, compute_case<IN1*>>(compute_case<RETURN_TYPE*>& ret, const IN0& a, const compute_case<IN1*>& b);\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN0*>, IN1>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const IN1& b)\

#define DECLARE_NATIVE_BINARY_COMMUTATIVE(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN0*>, compute_case<IN1*>>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const compute_case<IN1*>& b);\
template void UFUNC_NAME<compute_case<RETURN_TYPE*>, compute_case<IN0*>, IN1>(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const IN1& b)\

#define ADD_NATIVE_BINARY(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
};\
tables::UFUNC_NAME##_scalarLeft[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const IN0&, const compute_case<IN1*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
};\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
}

#define ADD_NATIVE_BINARY_COMMUTATIVE(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
};\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&)>(VA_UFUNC_MODULE::UFUNC_NAME))\
}

#define ADD_CAST_BINARY(UFUNC_NAME, MODEL_IN0, MODEL_IN1, IN0, IN1)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()];\
tables::UFUNC_NAME##_scalarLeft[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME##_scalarLeft[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()];\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()]

#define ADD_CAST_BINARY_COMMUTATIVE(UFUNC_NAME, MODEL_IN0, MODEL_IN1, IN0, IN1)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()];\
tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME##_scalarRight[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()]

#endif //VATENSOR_ARCH_UTIL_HPP
