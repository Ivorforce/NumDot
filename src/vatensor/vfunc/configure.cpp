#include "configure.hpp"

#include "vatensor/gen/base.hpp"
#include "ufunc_features.hpp"

void va::vfunc::configure() {
	va::vfunc::base::configure();
	// TODO
	// if (true) {
	// 	va::vfunc::avx2::configure();
	// }
	// if (true) {
	// 	va::vfunc::avx512::configure();
	// }
	// if (true) {
	// 	va::vfunc::sve::configure();
	// }
}

// Let us configure ourselves.
class Initializer { public: Initializer() {
	va::vfunc::configure();
}};

Initializer initializer;
