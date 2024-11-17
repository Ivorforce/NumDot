#ifndef VATENSOR_VIO_HPP
#define VATENSOR_VIO_HPP

#include "varray.hpp"
#include <memory>

namespace va {
	std::shared_ptr<VArray> load_npy(const char* data, std::size_t size);
	std::string save_npy(VData& data);
}

#endif //VATENSOR_VIO_HPP
