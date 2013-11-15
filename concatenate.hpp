/* CSCI435 Computer Vision
 * Project: Facial Recognition
 * Paul Foster - 3648370
 */
#ifndef CONCATENATE_HPP
#define CONCATENATE_HPP

#include <sstream>

// This is a tool to concatenate arbitrarily many, hetrogeneous types into a string.
// E.g. concatenate("CSC")('I')(435).str -> "CSCI435"
struct Concatenate
{
    std::string str;
    template <typename T>
    Concatenate(const T& t) :
        str()
    {
        (*this)(t);
    }
    template <typename T>
    Concatenate& operator()(const T& t)
    {
        std::stringstream ss;
        ss << str << t;
        str = ss.str();
        return *this;
    }
};

#endif
