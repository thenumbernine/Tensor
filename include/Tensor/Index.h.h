#pragma once

#include "Tensor/Meta.h"	//is_tensor_v

namespace Tensor {

//index-access classes
struct IndexBase;

template<char ident>
struct Index;

//forward-declare for index-access
template<typename InputTensorType, typename IndexVector>
requires is_tensor_v<InputTensorType>
struct IndexAccess;

}
