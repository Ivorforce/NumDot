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

static void add_native_unary(va::vfunc::tables::UFuncTableUnary& table, const va::DType out, const va::DType in0, void *function_ptr) {
	table[in0] = va::vfunc::UFunc<1> {
		{ in0 },
		out,
		function_ptr
	};
}

#define ADD_NATIVE_UNARY0(UFUNC_NAME, RETURN_TYPE, IN0)\
add_native_unary(\
	tables::UFUNC_NAME,\
	va::dtype_of_type<RETURN_TYPE>(),\
	va::dtype_of_type<IN0>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&)>(UFUNC_NAME))\
)

#define ADD_NATIVE_UNARY1(UFUNC_NAME, RETURN_TYPE, IN0, ARG1)\
add_native_unary(\
	tables::UFUNC_NAME,\
	va::dtype_of_type<RETURN_TYPE>(),\
	va::dtype_of_type<IN0>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, ARG1)>(UFUNC_NAME))\
)

static void add_cast(va::vfunc::tables::UFuncTableUnary& table, const va::DType model_in0, const va::DType in0) {
	table[in0] = table[model_in0];
}

#define ADD_CAST_UNARY(UFUNC_NAME, MODEL_IN0, IN0)\
add_cast(tables::UFUNC_NAME, va::dtype_of_type<MODEL_IN0>(), va::dtype_of_type<IN0>())

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

static void add_native_binary(va::vfunc::tables::UFuncTablesBinary& tables, const va::DType out, const va::DType in0, const va::DType in1, void* function_ptr_tensors, void* function_ptr_scalar_left, void* function_ptr_scalar_right) {
	tables.tensors[in0][in1] = va::vfunc::UFunc<2> {
		{ in0, in1 },
		out,
		function_ptr_tensors
	};
	tables.scalar_left[in0][in1] = va::vfunc::UFunc<2> {
		{ in0, in1 },
		out,
		function_ptr_scalar_left
	};
	tables.scalar_right[in0][in1] = va::vfunc::UFunc<2> {
		{ in0, in1 },
		out,
		function_ptr_scalar_right
	};
}

#define ADD_NATIVE_BINARY0(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
	add_native_binary(\
	tables::UFUNC_NAME,\
	va::dtype_of_type<RETURN_TYPE>(),\
	va::dtype_of_type<IN0>(),\
	va::dtype_of_type<IN1>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&)>(UFUNC_NAME)),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const IN0&, const compute_case<IN1*>&)>(UFUNC_NAME)),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&)>(UFUNC_NAME))\
)

static void add_native_binary(va::vfunc::tables::UFuncTablesBinaryCommutative& tables, const va::DType out, const va::DType in0, const va::DType in1, void *function_ptr_tensors, void *function_ptr_scalar_right) {
	tables.tensors[in0][in1] = va::vfunc::UFunc<2> {
		{ in0, in1 },
		out,
		function_ptr_tensors
	};
	tables.scalar_right[in0][in1] = va::vfunc::UFunc<2> {
		{ in0, in1 },
		out,
		function_ptr_scalar_right
	};
}

#define ADD_NATIVE_BINARY_COMMUTATIVE0(UFUNC_NAME, RETURN_TYPE, IN0, IN1)\
add_native_binary(\
	tables::UFUNC_NAME,\
	va::dtype_of_type<RETURN_TYPE>(),\
	va::dtype_of_type<IN0>(),\
	va::dtype_of_type<IN1>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&)>(UFUNC_NAME)),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&)>(UFUNC_NAME))\
)

#define ADD_NATIVE_BINARY_COMMUTATIVE3(UFUNC_NAME, RETURN_TYPE, IN0, IN1, ARG1, ARG2, ARG3)\
add_native_binary(\
	tables::UFUNC_NAME,\
	va::dtype_of_type<RETURN_TYPE>(),\
	va::dtype_of_type<IN0>(),\
	va::dtype_of_type<IN1>(),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const compute_case<IN1*>&, ARG1, ARG2, ARG3)>(UFUNC_NAME)),\
	reinterpret_cast<void*>(static_cast<void (*)(compute_case<RETURN_TYPE*>&, const compute_case<IN0*>&, const IN1&, ARG1, ARG2, ARG3)>(UFUNC_NAME))\
)

static void add_cast(va::vfunc::tables::UFuncTablesBinaryCommutative& tables, const va::DType model_in0, const va::DType model_in1, const va::DType in0, const va::DType in1) {
	tables.tensors[in0][in1] = tables.tensors[model_in0][model_in1];
	tables.scalar_right[in0][in1] = tables.scalar_right[model_in0][model_in1];
}

static void add_cast(va::vfunc::tables::UFuncTablesBinary& tables, const va::DType model_in0, const va::DType model_in1, const va::DType in0, const va::DType in1) {
	tables.tensors[in0][in1] = tables.tensors[model_in0][model_in1];
	tables.scalar_left[in0][in1] = tables.scalar_left[model_in0][model_in1];
	tables.scalar_right[in0][in1] = tables.scalar_right[model_in0][model_in1];
}

#define ADD_CAST_BINARY(UFUNC_NAME, MODEL_IN0, MODEL_IN1, IN0, IN1)\
add_cast(tables::UFUNC_NAME, va::dtype_of_type<MODEL_IN0>(), va::dtype_of_type<MODEL_IN1>(), va::dtype_of_type<IN0>(), va::dtype_of_type<IN1>())

#endif //VATENSOR_ARCH_UTIL_HPP
