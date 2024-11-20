#include "varray.hpp"

#include "xtensor_store.hpp"
#include <cstddef>                         // for size_t
#include <stdexcept>                       // for runtime_error
#include <type_traits>                     // for decay_t, common_type_t
#include "xscalar_store.hpp"
#include "xtensor/xstrided_view_base.hpp"  // for strided_view_args
#include "dtype.hpp"

const va::shape_type& va::shape(const VData& read) {
    return std::visit(
        [](const auto& carray) -> const va::shape_type& {
            return carray.shape();
        }, read
    );
}

const va::strides_type& va::strides(const VData& read) {
    return std::visit(
        [](const auto& carray) -> const va::strides_type& {
            return carray.strides();
        }, read
    );
}

va::size_type va::offset(const VData& read) {
    return std::visit(
        [](const auto& carray) -> va::size_type {
            return carray.data_offset();
        }, read
    );
}

xt::layout_type va::layout(const VData& read) {
    return std::visit(
        [](const auto& carray) -> xt::layout_type {
            return carray.layout();
        }, read
    );
}

std::size_t va::size(const VData& read) {
    return std::visit(
        [](const auto& carray) -> std::size_t {
            return carray.size();
        }, read
    );
}

std::size_t va::dimension(const VData& read) {
    return std::visit(
        [](const auto& carray) -> std::size_t {
            return carray.dimension();
        }, read
    );
}

va::VScalar va::to_single_value(const VData& read) {
    return std::visit(
        [](const auto& carray) -> va::VScalar {
            if (carray.size() != 1) {
                throw std::runtime_error("Expected a single element after slicing.");
            }
            return *carray.data();
            // TODO I expected this to work, but it doesn't. See https://xtensor.readthedocs.io/en/latest/indices.html#operator
            // But at least the above is a view, so no copy is made.
            // return V(array[slice]);
        }, read
    );
}

std::shared_ptr<va::VArray> va::VArray::sliced(const xt::xstrided_slice_vector& slices) const {
    return std::visit(
        [this, &slices](const auto& read) -> std::shared_ptr<VArray> {
            std::ptrdiff_t new_offset = 0;
            return std::make_shared<VArray>(
                VArray {
                    store,
                    slice_compute(read, slices, new_offset),
                    data_offset + new_offset
                }
            );
        }, data
    );
}

va::VData va::sliced_data(const VData& data, const xt::xstrided_slice_vector& slices) {
    return std::visit(
        [&slices](const auto& read) -> va::VData {
            std::ptrdiff_t new_offset = 0;  // Not needed
            return slice_compute(read, slices, new_offset);
        }, data
    );
}

va::VArray::operator bool() const { return va::static_cast_scalar<bool>(to_single_value()); }
va::VArray::operator int64_t() const { return va::static_cast_scalar<int64_t>(to_single_value()); }
va::VArray::operator int32_t() const { return va::static_cast_scalar<int32_t>(to_single_value()); }
va::VArray::operator int16_t() const { return va::static_cast_scalar<int16_t>(to_single_value()); }
va::VArray::operator int8_t() const { return va::static_cast_scalar<int8_t>(to_single_value()); }
va::VArray::operator uint64_t() const { return va::static_cast_scalar<uint64_t>(to_single_value()); }
va::VArray::operator uint32_t() const { return va::static_cast_scalar<uint32_t>(to_single_value()); }
va::VArray::operator uint16_t() const { return va::static_cast_scalar<uint16_t>(to_single_value()); }
va::VArray::operator uint8_t() const { return va::static_cast_scalar<uint8_t>(to_single_value()); }
va::VArray::operator double() const { return va::static_cast_scalar<double>(to_single_value()); }
va::VArray::operator float() const { return va::static_cast_scalar<float>(to_single_value()); }
