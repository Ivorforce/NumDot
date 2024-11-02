#include "tensor_fixed_store.hpp"

template struct numdot::XTensorFixedStore<int32_t, xt::xshape<2>>;
template struct numdot::XTensorFixedStore<int32_t, xt::xshape<3>>;
template struct numdot::XTensorFixedStore<int32_t, xt::xshape<4>>;
template struct numdot::XTensorFixedStore<real_t, xt::xshape<2>>;
template struct numdot::XTensorFixedStore<real_t, xt::xshape<3>>;
template struct numdot::XTensorFixedStore<real_t, xt::xshape<4>>;
#ifdef REAL_T_IS_DOUBLE
// For color
template struct numdot::XTensorFixedStore<float_t, xt::xshape<4>>;
#endif
