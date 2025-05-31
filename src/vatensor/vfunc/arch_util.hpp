#ifndef VATENSOR_ARCH_UTIL_HPP
#define VATENSOR_ARCH_UTIL_HPP

#define DECLARE_VFUNC(UFUNC_NAME)\
struct UFUNC_NAME {\
	template <typename RETURN_TYPE, typename... ARGS>\
	static void run(va::compute_case<RETURN_TYPE*>& ret, ARGS... args) {\
		va::vfunc::impl::UFUNC_NAME(ret, std::forward<ARGS>(args)...);\
	}\
}

template <typename C, typename RETURN_TYPE, typename... ARGS>
void add_native(va::vfunc::tables::UFuncTableInplace& table) {
	const auto out = va::dtype_of_type<RETURN_TYPE>();
	table[out] = (void *)&C::template run<RETURN_TYPE, ARGS...>;
}

template <typename C, typename RETURN_TYPE, typename IN0, typename... ARGS>
void add_native(va::vfunc::tables::UFuncTableInplaceBinary& table) {
	const auto out = va::dtype_of_type<RETURN_TYPE>();
	const auto in0 = va::dtype_of_type<IN0>();
	table[out][in0] = (void *)&C::template run<RETURN_TYPE, const va::compute_case<IN0*>&, ARGS...>;
}

template <typename C, typename RETURN_TYPE, typename IN0, typename... ARGS>
void add_native(va::vfunc::tables::UFuncTableUnary& table) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto out = va::dtype_of_type<RETURN_TYPE>();

	table[in0] = va::vfunc::VFunc<1> {
		{ in0 },
		out,
		(void *)&C::template run<RETURN_TYPE, const va::compute_case<IN0*>&, ARGS...>
	};
}

template <typename C, typename RETURN_TYPE, typename IN0, typename IN1, typename... ARGS>
void add_native(va::vfunc::tables::UFuncTablesBinary& tables) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto in1 = va::dtype_of_type<IN1>();
	const auto out = va::dtype_of_type<RETURN_TYPE>();

	tables.tensors[in0][in1] = va::vfunc::VFunc<2> {
		{ in0, in1 },
		out,
		(void *)&C::template run<RETURN_TYPE, const va::compute_case<IN0*>&, const va::compute_case<IN1*>&, ARGS...>
	};
	tables.scalar_left[in0][in1] = va::vfunc::VFunc<2> {
		{ in0, in1 },
		out,
		(void *)&C::template run<RETURN_TYPE, const IN0&, const va::compute_case<IN1*>&, ARGS...>
	};
	tables.scalar_right[in0][in1] = va::vfunc::VFunc<2> {
		{ in0, in1 },
		out,
		(void *)&C::template run<RETURN_TYPE, const va::compute_case<IN0*>&, const IN1&, ARGS...>
	};
}

template <typename C, typename RETURN_TYPE, typename IN0, typename IN1, typename... ARGS>
void add_native(va::vfunc::tables::UFuncTablesBinaryCommutative& tables) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto in1 = va::dtype_of_type<IN1>();
	const auto out = va::dtype_of_type<RETURN_TYPE>();

	tables.tensors[in0][in1] = va::vfunc::VFunc<2> {
		{ in0, in1 },
		out,
		(void *)&C::template run<RETURN_TYPE, const va::compute_case<IN0*>&, const va::compute_case<IN1*>&, ARGS...>
	};
	tables.scalar_right[in0][in1] = va::vfunc::VFunc<2> {
		{ in0, in1 },
		out,
		(void *)&C::template run<RETURN_TYPE, const va::compute_case<IN0*>&, const IN1&, ARGS...>
	};
}

template <typename C, typename RETURN_TYPE, typename IN0, typename IN1, typename... ARGS>
void add_native(va::vfunc::tables::UFuncTableBinary& table) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto in1 = va::dtype_of_type<IN1>();
	const auto out = va::dtype_of_type<RETURN_TYPE>();

	table[in0][in1] = va::vfunc::VFunc<2> {
		{ in0, in1 },
		out,
		(void *)&C::template run<RETURN_TYPE, const va::compute_case<IN0*>&, const va::compute_case<IN1*>&, ARGS...>
	};
}

template <typename IN0, typename MODEL_IN0>
void add_cast(va::vfunc::tables::UFuncTableUnary& table) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto model_in0 = va::dtype_of_type<MODEL_IN0>();
	table[in0] = table[model_in0];
}

template <typename IN0, typename IN1, typename MODEL_IN0, typename MODEL_IN1>
void add_cast(va::vfunc::tables::UFuncTablesBinary& tables) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto in1 = va::dtype_of_type<IN1>();
	const auto model_in0 = va::dtype_of_type<MODEL_IN0>();
	const auto model_in1 = va::dtype_of_type<MODEL_IN1>();

	tables.tensors[in0][in1] = tables.tensors[model_in0][model_in1];
	tables.scalar_left[in0][in1] = tables.scalar_left[model_in0][model_in1];
	tables.scalar_right[in0][in1] = tables.scalar_right[model_in0][model_in1];
}

template <typename IN0, typename IN1, typename MODEL_IN0, typename MODEL_IN1>
void add_cast(va::vfunc::tables::UFuncTablesBinaryCommutative& tables) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto in1 = va::dtype_of_type<IN1>();
	const auto model_in0 = va::dtype_of_type<MODEL_IN0>();
	const auto model_in1 = va::dtype_of_type<MODEL_IN1>();

	tables.tensors[in0][in1] = tables.tensors[model_in0][model_in1];
	tables.scalar_right[in0][in1] = tables.scalar_right[model_in0][model_in1];
}

template <typename IN0, typename IN1, typename MODEL_IN0, typename MODEL_IN1>
void add_cast(va::vfunc::tables::UFuncTableBinary& table) {
	const auto in0 = va::dtype_of_type<IN0>();
	const auto in1 = va::dtype_of_type<IN1>();
	const auto model_in0 = va::dtype_of_type<MODEL_IN0>();
	const auto model_in1 = va::dtype_of_type<MODEL_IN1>();

	table[in0][in1] = table[model_in0][model_in1];
}

#endif //VATENSOR_ARCH_UTIL_HPP
