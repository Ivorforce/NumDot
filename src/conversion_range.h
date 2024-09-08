#ifndef CONVERSION_RANGE_H
#define CONVERSION_RANGE_H

range_part to_range_part(const Variant& variant) {
    switch (variant.get_type()) {
        case Variant::INT:
            return int64_t(variant);
        case NULL:
            return xt::placeholders::xtuph{};
        default:
            throw std::runtime_error("Invalid type for range.");
    }
}

#endif //CONVERSION_RANGE_H
