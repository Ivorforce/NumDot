#ifndef CONVERSION_STRING_H
#define CONVERSION_STRING_H

#include "godot_cpp/variant/string.hpp"       // for String
#include "xtensor/xio.hpp"                    // for xt expression to string

// From https://github.com/xtensor-stack/xtensor/issues/1413
template <typename E>
String xt_to_string(E&& e)
{
    std::ostringstream out;
    out << std::forward<E>(e);
    return String(out.str().c_str());
}

#endif //CONVERSION_STRING_H
