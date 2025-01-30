// Dirty hack to avoid defining the tables twice.

#define UNARY_TABLES(UFUNC_NAME) UFuncTableUnary UFUNC_NAME;
#define BINARY_TABLES(UFUNC_NAME) UFuncTablesBinary UFUNC_NAME;
#define BINARY_TABLES_COMMUTATIVE(UFUNC_NAME) UFuncTablesBinaryCommutative UFUNC_NAME;

#include "tables.hpp"
