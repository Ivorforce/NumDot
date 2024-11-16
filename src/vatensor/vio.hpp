#ifndef VATENSOR_VIO_HPP
#define VATENSOR_VIO_HPP

#include "varray.hpp"
#include <memory>

namespace va {
	std::shared_ptr<VArray> load_npy(char* data, std::size_t size);
}

#endif //VATENSOR_VIO_HPP
