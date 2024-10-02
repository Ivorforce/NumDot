#ifndef ROUND_H
#define ROUND_H

#include "auto_defines.h"
#include "varray.h"

namespace va {
	void ceil(VArrayTarget target, const VArray& array);
	void floor(VArrayTarget target, const VArray& array);
	void trunc(VArrayTarget target, const VArray& array);
	void round(VArrayTarget target, const VArray& array);
	void nearbyint(VArrayTarget target, const VArray& array);
}

#endif //ROUND_H
