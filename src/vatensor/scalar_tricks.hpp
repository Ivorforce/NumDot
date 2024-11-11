#ifndef SCALAR_TRICKS_HPP
#define SCALAR_TRICKS_HPP

#define OPTIMIZE_COMMUTATIVE(FN_NAME, ALLOCATOR, TARGET, A_ARRAY, B_ARRAY)\
if (va::dimension(A_ARRAY) == 0) {\
	FN_NAME(ALLOCATOR, TARGET, B_ARRAY, va::to_single_value(A_ARRAY));\
	return;\
}\
if (va::dimension(B_ARRAY) == 0) {\
	FN_NAME(ALLOCATOR, TARGET, A_ARRAY, va::to_single_value(B_ARRAY));\
	return;\
}

#define OPTIMIZE_NONCOMMUTATIVE(FN_NAME, ALLOCATOR, TARGET, A_ARRAY, B_ARRAY)\
if (va::dimension(A_ARRAY) == 0) {\
	FN_NAME(ALLOCATOR, TARGET, va::to_single_value(A_ARRAY), B_ARRAY);\
	return;\
}\
if (va::dimension(B_ARRAY) == 0) {\
	FN_NAME(ALLOCATOR, TARGET, A_ARRAY, va::to_single_value(B_ARRAY));\
	return;\
}

#define OPTIMIZE_COMMUTATIVE_REDUCTION(FN_NAME, A_ARRAY, B_ARRAY)\
if (va::dimension(A_ARRAY) == 0) {\
	return FN_NAME(B_ARRAY, va::to_single_value(A_ARRAY));\
}\
if (va::dimension(B_ARRAY) == 0) {\
	return FN_NAME(A_ARRAY, va::to_single_value(B_ARRAY));\
}

#endif //SCALAR_TRICKS_HPP
