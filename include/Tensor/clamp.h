#pragma once

#include "Tensor/Tensor.h"
#include <algorithm>	//min, max

namespace Tensor {

template<typename T>
constexpr decltype(auto) clamp(T x, T min, T max) {
	return std::min(max, std::max(min, x));
}

template<typename T>
requires (Tensor::is_tensor_v<T>)
T clamp(T const &x, T const &min, T const &max) {
	return T([&](typename T::intN i) -> typename T::Scalar {
		return clamp<typename T::Scalar>(x(i), min(i), max(i));
	});
}

}
