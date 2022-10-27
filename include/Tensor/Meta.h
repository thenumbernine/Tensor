#pragma once

#include "Common/Meta.h"

namespace Tensor {

// TODO move to Common/Meta.h
// TODO how about just a template<typename,typename> apply_all ?
// then is_all_base_of_v<T,Ts...> = apply_all<std::is_base_of, T, Ts...>;
template<typename T, typename... Us>
struct is_all_base_of;
template<typename T, typename U, typename... Us>
struct is_all_base_of<T,U,Us...> {
	static constexpr bool value = std::is_base_of_v<T,U>
		&& is_all_base_of<T, Us...>::value;
};
template<typename T>
struct is_all_base_of<T> {
	static constexpr bool value = true;
};
template<typename T, typename... Us>
concept is_all_base_of_v = is_all_base_of<T, Us...>::value;


/*
Detects if a class is a "tensor".
These include _vec _sym _asym and subclasses (like _quat).
It's defined in the class in the TENSOR_HEADER, as a static constexpr field.

TODO should this decay_t T or should I rely on the invoker to is_tensor_v<decay_t<T>> ?
*/
template<typename T>
concept is_tensor_v = T::isTensorFlag;

}
