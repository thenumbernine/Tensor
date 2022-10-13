#pragma once

namespace Tensor {

// these are in Inverse.h:
// they are giving me link errors.
// do I have to forward-declare all template partial specializations to fix it?
// idk, never fixed it, just avoid using .determinant() instead

template<typename T>
requires is_tensor_v<T>
inline typename T::Scalar determinant(T const & a);

template<typename T>
requires is_tensor_v<T>
inline T inverse(
	T const & a,
	typename T::Scalar const & det
);

template<typename T>
requires is_tensor_v<T>
inline T inverse(T const & a);

}
