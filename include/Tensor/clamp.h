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

// TODO clamp(Tensor, ScalarType, ScalarType) also?
template<typename T>
requires (Tensor::is_tensor_v<T>)
T clamp(T const &x, T const &min, T const &max) {
	return T([&](typename T::intN i) -> typename T::ScalarType {
		return clamp<typename T::ScalarType>(x(i), min(i), max(i));
	});
}

}
