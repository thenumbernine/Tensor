#pragma once

#include "Tensor/Tensor.h"
#include <algorithm>	//min, max

namespace Tensor {

template<typename T>
T clamp(T x, T xmin, T xmax) {
	return std::min(xmax, std::max(xmin, x));
}

template<typename T>
requires (Tensor::is_tensor_v<T>)
T clamp(T const & x, T const & xmin, T const & xmax) {
	return T([&](typename T::intN i) -> typename T::Scalar {
		return clamp<typename T::Scalar>(x(i), xmin(i), xmax(i));
	});
}

}
