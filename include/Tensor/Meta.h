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
constexpr bool is_tensor_v = requires(T const & t) { &T::isTensorFlag; };

//https://stackoverflow.com/a/48840842
// begin static_for

// std::size is supported from C++17
// TODO compare this with the variadic and sequence loops already in Common
// and TODO see if any of them can resolve template args ... 

template <typename T, size_t N>
constexpr size_t static_size(const T (&)[N]) noexcept {
	return N;
}

template <typename ...T>
constexpr size_t static_size(const std::tuple<T...> &) {
	return std::tuple_size<std::tuple<T...> >::value;
}

template<typename Functor>
void runtime_for_lt(Functor && function, size_t from, size_t to) {
	if (from < to) {
		function(from);
		runtime_for_lt(std::forward<Functor>(function), from + 1, to);
	}
}

template <template <typename T_> class Functor, typename T>
void runtime_foreach(T & container) {
	runtime_for_lt(Functor<T>{ container }, 0, static_size(container));
}

template <typename Functor, typename T>
void runtime_foreach(T & container, Functor && functor) {
	runtime_for_lt(functor, 0, static_size(container));
}

template <typename T>
void static_consume(std::initializer_list<T>) {}

template<typename Functor, std::size_t... S>
constexpr void static_foreach_seq(Functor && function, std::index_sequence<S...>) {
	return static_consume({ (function(std::integral_constant<std::size_t, S>{}), 0)... });
}

template<std::size_t Size, typename Functor>
constexpr void static_foreach(Functor && functor) {
	return static_foreach_seq(std::forward<Functor>(functor), std::make_index_sequence<Size>());
}

// end static_for

}
