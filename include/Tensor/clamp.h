#pragma once

#include "Tensor/Tensor.h"
#include <algorithm>	//min, max

namespace Tensor {

template<typename T>
T clamp(T x, T xmin, T xmax) {
	if constexpr(Tensor::is_tensor_v<T>){
		return T([&](typename T::intN i) -> typename T::Scalar {
			return clamp<typename T::Scalar>(x(i), xmin(i), xmax(i));
		});
	} else {
		return std::min(xmax, std::max(xmin, x));
	}
}

}
