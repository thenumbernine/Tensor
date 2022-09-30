#pragma once

#include "Tensor/Tensor.h"

namespace Tensor {

template<typename T>
T& clamp(T &x, T &min, T &max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

template<typename T>
T const& clamp(T const &x, T const &min, T const &max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

template<typename T>
requires (Tensor::is_tensor_v<T>)
T clamp(T const &x, T const &min, T const &max) {
	return T([&](typename T::intN i) -> typename T::Scalar {
		return clamp<typename T::Scalar>(x(i), min(i), max(i));
	});
}

}
