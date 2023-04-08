#pragma once

#include "Tensor/Vector.h.h"

namespace Tensor {

//forward-declare, body is below all tensor classes.

template<typename T>
requires is_tensor_v<T>
T elemMul(T const & a, T const & b);

template<typename... T>
auto matrixCompMult(T&&... args);

template<typename... T>
auto hadamard(T&&... args);

template<typename A, typename B>
requires IsBinaryTensorOp<A, B>
typename A::Scalar inner(A const & a, B const & b);

template<typename... T>
auto dot(T&&... args);

template<typename T> requires is_tensor_v<T>
typename T::Scalar lenSq(T const & v);

template<typename T> requires (is_tensor_v<T>)
typename T::Scalar length(T const & v);

template<typename A, typename B>
requires IsBinaryTensorOp<A, B>
typename A::Scalar distance(A const & a, B const & b);

template<typename T>
requires (is_tensor_v<T>)
T normalize(T const & v);

template<typename A, typename B>
requires IsBinaryTensorR3xR3Op<A, B>
auto cross(A const & a, B const & b);

template<typename A, typename B>
auto outer(A const & a, B const & b);

template<typename... T>
auto outerProduct(T&&... args);

template<int m=0, int n=1, typename T>
requires (
	is_tensor_v<T>
	&& T::rank >= 2
)
auto transpose(T const & t);

template<int m=0, int n=1, typename T>
requires (is_tensor_v<T>
	&& m < T::rank
	&& n < T::rank
	&& T::template dim<m> == T::template dim<n>
)
auto contract(T const & t);

template<int m=0, int n=1, typename T>
auto trace(T const & o);

template<int index=0, int count=1, typename A>
requires (is_tensor_v<A>)
auto contractN(A const & a);

template<int num=1, typename A, typename B>
requires IsInteriorOp<num, A, B>
auto interior(A const & a, B const & b);

template<int m=0, typename T>
requires (is_tensor_v<T>)
auto diagonal(T const & t);

template<typename T>
requires IsSquareTensor<T>
auto makeSym(T const & t);

template<typename T>
requires IsSquareTensor<T>
auto makeAsym(T const & t);

template<typename A, typename B>
auto wedge(A const & a, B const & b);

template<typename T>
requires IsSquareTensor<T>
auto hodgeDual(T const & a);

//name
template<typename... T>
auto dual(T&&... args);

//wedge all rows of a m x n matrix
template<int i = 0>
auto wedgeAll(auto const & v);

template<typename A, typename B>
requires IsBinaryTensorOpWithMatchingNeighborDims<A, B>
auto operator*(A const & a, B const & b);

//funny, 'if constexpr' causes this to lose constexpr-ness, but ternary is ok.
constexpr int constexpr_isqrt_r(int inc, int limit) {
	return inc * inc > limit ? inc-1 : constexpr_isqrt_r(inc+1, limit);
}
constexpr int constexpr_isqrt(int i) {
	return constexpr_isqrt_r(0, i);
}

//https://en.cppreference.com/w/cpp/language/constexpr
constexpr int constexpr_factorial(int n) {
	return n <= 1 ? 1 : (n * constexpr_factorial(n-1));
}
constexpr int consteval_nChooseR(int m, int n) {
    return constexpr_factorial(n) / constexpr_factorial(m) / constexpr_factorial(n - m);
}

//https://stackoverflow.com/a/9331125
constexpr int nChooseR(int n, int k) {
    if (k > n) return 0;
    if (k << 1 > n) k = n - k;
    if (k == 0) return 1;
    int result = n;
    // TODO can you guarantee that /=i will always have 'i' as a divisor? or do we need two loops?
	for (int i = 2; i <= k; ++i) {
		result *= n - i + 1;
		result /= i;
    }
    return result;
}

}
