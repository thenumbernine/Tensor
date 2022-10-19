#pragma once

#include "Tensor/Meta.h"	//is_tensor_v
#include <utility>			//integer_sequence
#include <tuple>			//tuple

namespace Tensor {

//index-access classes
struct IndexBase;

template<char ident>
struct Index;

//forward-declare for tensors's operator() used for index-notation
template<typename InputTensorType, typename IndexVector>
requires is_tensor_v<InputTensorType>
struct IndexAccess;

//forward-declare cuz operator() IndexAccess needs to know if its making a scalar or tensor
template<typename IndexTuple_>
struct IndexAccessDetails;

}
