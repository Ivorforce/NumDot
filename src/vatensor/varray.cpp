#include "varray.hpp"

#include "xarray_store.hpp"
#include <cstddef>                         // for size_t
#include <stdexcept>                       // for runtime_error
#include <type_traits>                     // for decay_t, common_type_t
#include "xtensor/xstrided_view_base.hpp"  // for strided_view_args

const va::shape_type& va::shape(const VRead& read) {
    return std::visit(
        [](const auto& carray) -> const va::shape_type& {
            return carray.shape();
        }, read
    );
}

const va::strides_type& va::strides(const VRead& read) {
    return std::visit(
        [](const auto& carray) -> const va::strides_type& {
            return carray.strides();
        }, read
    );
}

va::size_type va::offset(const VRead& read) {
    return std::visit(
        [](const auto& carray) -> va::size_type {
            return carray.data_offset();
        }, read
    );
}

xt::layout_type va::layout(const VRead& read) {
    return std::visit(
        [](const auto& carray) -> xt::layout_type {
            return carray.layout();
        }, read
    );
}

va::DType va::dtype(const VRead& read) {
    return static_cast<DType>(read.index());
}

std::size_t va::size(const VRead& read) {
    return std::visit(
        [](const auto& carray) -> std::size_t {
            return carray.size();
        }, read
    );
}

std::size_t va::dimension(const VRead& read) {
    return std::visit(
        [](const auto& carray) -> std::size_t {
            return carray.dimension();
        }, read
    );
}

std::size_t va::size_of_array_in_bytes(const VRead& read) {
    return std::visit(
        [](auto& carray) {
            using V = typename std::decay_t<decltype(carray)>::value_type;
            return carray.size() * sizeof(V);
        }, read
    );
}

va::VScalar va::to_single_value(const VRead& read) {
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

void va::VArray::prepare_write() {
    if (write.has_value()) {
        return;
    }

    write = store->make_write(read);
}

std::shared_ptr<va::VArray> va::VArray::sliced(const xt::xstrided_slice_vector& slices) const {
    return std::visit(
        [this, &slices](const auto& read) -> std::shared_ptr<VArray> {
            return std::make_shared<VArray>(
                VArray {
                    store,
                    slice_compute(read, slices),
                    {}
                }
            );
        }, read
    );
}

va::VRead va::VArray::sliced_read(const xt::xstrided_slice_vector& slices) const {
    return std::visit(
        [&slices](const auto& read) -> va::VRead {
            return slice_compute(read, slices);
        }, read
    );
}

va::VWrite va::VArray::sliced_write(const xt::xstrided_slice_vector& slices) {
    prepare_write();

    return std::visit(
        [&slices](auto& write) -> va::VWrite {
            return slice_compute(write, slices);
        }, write.value()
    );
}

va::VScalar va::dtype_to_variant(const DType dtype) {
    switch (dtype) {
        case DType::Bool:
            return bool();
        case DType::Float32:
            return float_t();
        case DType::Float64:
            return double_t();
        case DType::Complex64:
            return std::complex<float_t>();
        case DType::Complex128:
            return std::complex<double_t>();
        case DType::Int8:
            return int8_t();
        case DType::Int16:
            return int16_t();
        case DType::Int32:
            return int32_t();
        case DType::Int64:
            return int64_t();
        case DType::UInt8:
            return uint8_t();
        case DType::UInt16:
            return uint16_t();
        case DType::UInt32:
            return uint32_t();
        case DType::UInt64:
            return int64_t();
        default:
            throw std::runtime_error("Invalid dtype.");
    }
}

va::DType va::variant_to_dtype(const VScalar dtype) {
    return static_cast<DType>(dtype.index());
}

std::size_t va::size_of_dtype_in_bytes(const DType dtype) {
    return std::visit(
        [](auto dtype) {
            return sizeof(dtype);
        }, dtype_to_variant(dtype)
    );
}

va::VScalar va::scalar_to_dtype(const VScalar v, const DType dtype) {
    return std::visit(
        [v](const auto t) -> va::VScalar {
            using T = std::decay_t<decltype(t)>;
            return va::scalar_to_type<T>(v);
        }, dtype_to_variant(dtype)
    );
}

va::DType va::dtype_common_type(const DType a, const DType b) {
    if (a == DTypeMax) return b;
    if (b == DTypeMax) return a;

    return std::visit(
        [](auto a, auto b) { return variant_to_dtype(std::common_type_t<decltype(a), decltype(b)>()); },
        dtype_to_variant(a),
        dtype_to_variant(b)
    );
}

va::VArray::operator bool() const { return va::scalar_to_type<bool>(to_single_value()); }
va::VArray::operator int64_t() const { return va::scalar_to_type<int64_t>(to_single_value()); }
va::VArray::operator int32_t() const { return va::scalar_to_type<int32_t>(to_single_value()); }
va::VArray::operator int16_t() const { return va::scalar_to_type<int16_t>(to_single_value()); }
va::VArray::operator int8_t() const { return va::scalar_to_type<int8_t>(to_single_value()); }
va::VArray::operator uint64_t() const { return va::scalar_to_type<uint64_t>(to_single_value()); }
va::VArray::operator uint32_t() const { return va::scalar_to_type<uint32_t>(to_single_value()); }
va::VArray::operator uint16_t() const { return va::scalar_to_type<uint16_t>(to_single_value()); }
va::VArray::operator uint8_t() const { return va::scalar_to_type<uint8_t>(to_single_value()); }
va::VArray::operator double() const { return va::scalar_to_type<double>(to_single_value()); }
va::VArray::operator float() const { return va::scalar_to_type<float>(to_single_value()); }

std::shared_ptr<va::VArray> va::from_scalar_variant(VScalar scalar) {
    return std::visit(
        [](auto cscalar) {
            return va::store::from_scalar(cscalar);
        }, scalar
    );
}
