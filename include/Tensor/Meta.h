#pragma once

#include "Common/Meta.h"

namespace Tensor {

/*
Detects if a class is a "tensor".
These include _vec _sym _asym and subclasses (like _quat).
It's defined in the class in the TENSOR_HEADER, as a static constexpr field.

TODO should this decay_t T or should I rely on the invoker to is_tensor_v<decay_t<T>> ?
*/
template<typename T>
concept is_tensor_v = T::isTensorFlag;

}
