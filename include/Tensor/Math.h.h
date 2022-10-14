#pragma once

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
requires IsBinaryTensorR3xR3Op<A,B>
auto cross(A const & a, B const & b);

template<typename A, typename B>
requires IsBinaryTensorOp<A, B>
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
requires IsBinaryTensorOp<A,B>
auto wedge(A const & a, B const & b);

template<typename T>
requires IsSquareTensor<T>
auto hodgeDual(T const & a);

template<typename A, typename B>
requires IsBinaryTensorOpWithMatchingNeighborDims<A,B>
auto operator*(A const & a, B const & b);

}
