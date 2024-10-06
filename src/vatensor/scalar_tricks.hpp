#ifndef SCALAR_TRICKS_HPP
#define SCALAR_TRICKS_HPP

#define OPTIMIZE_COMMUTATIVE(FN_NAME, A_ARRAY, B_ARRAY)\
if (A_ARRAY.dimension() == 0) {\
	FN_NAME(target, B_ARRAY, A_ARRAY.to_single_value());\
	return;\
}\
if (B_ARRAY.dimension() == 0) {\
	FN_NAME(target, A_ARRAY, B_ARRAY.to_single_value());\
	return;\
}

#endif //SCALAR_TRICKS_HPP
