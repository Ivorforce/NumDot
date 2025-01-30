#ifndef VATENSOR_ARCH_UTIL_HPP
#define VATENSOR_ARCH_UTIL_HPP

#define DECLARE_NATIVE_UNARY0(UFUNC_NAME, RETURN_TYPE, IN0)\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a) {\
	va::vfunc::impl::UFUNC_NAME(ret, a);\
}

#define DECLARE_NATIVE_UNARY1(UFUNC_NAME, RETURN_TYPE, IN0, ARG1)\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, ARG1 arg1) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, arg1);\
}

#define ADD_NATIVE_UNARY0(UFUNC_NAME, RETURN_TYPE, IN0)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()] = UFunc<1> {\
	{ va::dtype_of_type<IN0>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&)>(UFUNC_NAME))\
}

#define ADD_NATIVE_UNARY1(UFUNC_NAME, RETURN_TYPE, IN0, ARG1)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()] = UFunc<1> {\
	{ va::dtype_of_type<IN0>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, ARG1)>(UFUNC_NAME))\
}

#define ADD_CAST_UNARY(UFUNC_NAME, MODEL_IN0, IN0)\
tables::UFUNC_NAME[va::dtype_of_type<IN0>()] = tables::UFUNC_NAME[va::dtype_of_type<MODEL_IN0>()]


#define DECLARE_NATIVE_BINARY0(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const compute_case<IN1*>& b) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, b);\
}\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const IN0& a, const compute_case<IN1*>& b) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, b);\
}\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const IN1& b) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, b);\
}

#define DECLARE_NATIVE_BINARY_COMMUTATIVE0(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const compute_case<IN1*>& b) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, b);\
}\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const IN1& b) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, b);\
}

#define DECLARE_NATIVE_BINARY_COMMUTATIVE3(UFUNC_NAME, RETURN_TYPE, IN0, IN1, ARG1, ARG2, ARG3)\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const compute_case<IN1*>& b, ARG1 arg1, ARG2 arg2, ARG3 arg3) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, b, arg1, arg2, arg3);\
}\
void UFUNC_NAME(compute_case<RETURN_TYPE*>& ret, const compute_case<IN0*>& a, const IN1& b, ARG1 arg1, ARG2 arg2, ARG3 arg3) {\
	va::vfunc::impl::UFUNC_NAME(ret, a, b, arg1, arg2, arg3);\
}

#define ADD_NATIVE_BINARY0(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
tables::UFUNC_NAME.tensors[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&)>(UFUNC_NAME))\
};\
tables::UFUNC_NAME.scalar_left[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const IN0&, const compute_case<IN1*>&)>(UFUNC_NAME))\
};\
tables::UFUNC_NAME.scalar_right[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&)>(UFUNC_NAME))\
}

#define ADD_NATIVE_BINARY_COMMUTATIVE0(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
tables::UFUNC_NAME.tensors[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&)>(UFUNC_NAME))\
};\
tables::UFUNC_NAME.scalar_right[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&)>(UFUNC_NAME))\
}

#define ADD_NATIVE_BINARY_COMMUTATIVE3(UFUNC_NAME, RETURN_TYPE, IN0, IN1, ARG1, ARG2, ARG3)\
tables::UFUNC_NAME.tensors[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&, ARG1, ARG2, ARG3)>(UFUNC_NAME))\
};\
tables::UFUNC_NAME.scalar_right[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = UFunc<2> {\
	{ va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>() },\
	va::dtype_of_type<RETURN_TYPE>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&, ARG1, ARG2, ARG3)>(UFUNC_NAME))\
}

#define ADD_CAST_BINARY(UFUNC_NAME, MODEL_IN0, MODEL_IN1, IN0, IN1)\
tables::UFUNC_NAME.tensors[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME.tensors[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()];\
tables::UFUNC_NAME.scalar_left[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME.scalar_left[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()];\
tables::UFUNC_NAME.scalar_right[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME.scalar_right[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()]

#define ADD_CAST_BINARY_COMMUTATIVE(UFUNC_NAME, MODEL_IN0, MODEL_IN1, IN0, IN1)\
tables::UFUNC_NAME.tensors[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME.tensors[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()];\
tables::UFUNC_NAME.scalar_right[va::dtype_of_type<IN0>()][va::dtype_of_type<IN1>()] = tables::UFUNC_NAME.scalar_right[va::dtype_of_type<MODEL_IN0>()][va::dtype_of_type<MODEL_IN1>()]

#endif //VATENSOR_ARCH_UTIL_HPP
