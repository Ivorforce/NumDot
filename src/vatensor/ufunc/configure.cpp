#include "configure.hpp"

#include "vatensor/gen/base.hpp"
#include "ufunc_features.hpp"

void va::ufunc::configure() {
	va::ufunc::base::configure();
	// TODO
	// if (true) {
	// 	va::ufunc::avx2::configure();
	// }
	// if (true) {
	// 	va::ufunc::avx512::configure();
	// }
	// if (true) {
	// 	va::ufunc::sve::configure();
	// }
}

// Let us configure ourselves.
class Initializer { public: Initializer() {
	va::ufunc::configure();
}};

Initializer initializer;
