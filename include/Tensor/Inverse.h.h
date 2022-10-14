#pragma once

#include "Tensor/Meta.h"	//is_tensor_v

namespace Tensor {

// these are in Inverse.h:

template<typename T>
requires is_tensor_v<T>
typename T::Scalar determinant(T const & a);


template<typename T>
requires is_tensor_v<T>
T inverse(T const & a, typename T::Scalar const & det);

template<typename T>
requires is_tensor_v<T>
T inverse(T const & a);

}
