#ifndef LOGICAL_H
#define LOGICAL_H

#include "auto_defines.hpp"
#include "varray.hpp"

namespace va {
	void logical_and(VArrayTarget target, const VArray& a, const VArray& b);
	void logical_or(VArrayTarget target, const VArray& a, const VArray& b);
	void logical_xor(VArrayTarget target, const VArray& a, const VArray& b);
	void logical_not(VArrayTarget target, const VArray& a);
}

#endif //LOGICAL_H
