#ifndef CONVERSION_STRING_H
#define CONVERSION_STRING_H

#include "godot_cpp/variant/string.hpp"       // for String
#include "xtensor/xio.hpp"                    // for xt expression to string

// From https://github.com/xtensor-stack/xtensor/issues/1413
template <typename E>
godot::String xt_to_string(E&& e)
{
    std::ostringstream out;
    out << std::forward<E>(e);
    return { out.str().c_str() };
}

template<typename T>
godot::String array_to_string(const T& arr) {
    std::ostringstream out;
    out << "[";
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        if (it != arr.begin()) {
            out << ", ";
        }
        out << *it;
    }
    out << "]";
    return { out.str().c_str() };
}

#endif //CONVERSION_STRING_H
