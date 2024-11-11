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

#endif //SCALAR_TRICKS_HPP
