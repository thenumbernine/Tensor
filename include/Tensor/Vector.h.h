#pragma once

// I have so many templates that I need to forward-declare them before using them...
#include "Tensor/Meta.h"	//is_tensor_v

namespace Tensor {

// concepts

template<typename A, typename B>
concept IsBinaryTensorOp =
	is_tensor_v<A>
	&& is_tensor_v<B>
//	&& std::is_same_v<typename A::Scalar, typename B::Scalar>	// TODO meh?
;

template<typename T>
concept IsSquareTensor =
	is_tensor_v<T>
	&& T::isSquare;

template<typename A, typename B>
concept IsBinaryTensorOpWithMatchingNeighborDims =
	IsBinaryTensorOp<A,B>
	&& A::template dim<A::rank-1> == B::template dim<0>;

template<typename A, typename B>
concept IsBinaryTensorR3xR3Op =
	IsBinaryTensorOp<A,B>
	// can't use _vec<int,1> because it hasn't been declared yet
	//&& A::dims == _vec<int,1>(3)
	//&& B::dims == _vec<int,1>(3);
	&& A::rank == 1 && A::template dim<0> == 3
	&& B::rank == 1 && B::template dim<0> == 3;

template<typename A, typename B>
concept IsBinaryTensorDiffTypeButMatchingDims =
	IsBinaryTensorOp<A,B>
	&& !std::is_same_v<A,B>
	&& A::dims() == B::dims(); // equal types means we use .operator== which is constexpr

template<int num, typename A, typename B>
concept IsInteriorOp =
	IsBinaryTensorOp<A,B> && num > 0 && num <= A::rank && num <= B::rank;
// TODO also assert the last 'num' dims of A match the first 'num' dims of B

//forward-declare everything

template<typename Inner, int localDim> struct _vec;
template<typename Inner, int localDim> struct _ident;
template<typename Inner, int localDim> struct _sym;
template<typename Inner, int localDim> struct _asym;
template<typename Inner, int localDim, int localRank> struct _symR;
template<typename Inner, int localDim, int localRank> struct _asymR;

//convention?  row-major to match math indexing, easy C inline ctor,  so A_ij = A[i][j]
// ... but OpenGL getFloatv(GL_...MATRIX) uses column-major so uploads have to be transposed
// ... also GLSL is column-major so between this and GLSL the indexes have to be transposed.
template<typename T, int dim1, int dim2> using _mat = _vec<_vec<T, dim2>, dim1>;


// specific-sized templates
template<typename T> using _vec2 = _vec<T,2>;
template<typename T> using _vec3 = _vec<T,3>;
template<typename T> using _vec4 = _vec<T,4>;
template<typename T> using _mat2x2 = _vec2<_vec2<T>>;
template<typename T> using _mat2x3 = _vec2<_vec3<T>>;
template<typename T> using _mat2x4 = _vec2<_vec4<T>>;
template<typename T> using _mat3x2 = _vec3<_vec2<T>>;
template<typename T> using _mat3x3 = _vec3<_vec3<T>>;
template<typename T> using _mat3x4 = _vec3<_vec4<T>>;
template<typename T> using _mat4x2 = _vec4<_vec2<T>>;
template<typename T> using _mat4x3 = _vec4<_vec3<T>>;
template<typename T> using _mat4x4 = _vec4<_vec4<T>>;
template<typename T> using _sym2 = _sym<T,2>;
template<typename T> using _sym3 = _sym<T,3>;
template<typename T> using _sym4 = _sym<T,4>;
template<typename T> using _asym2 = _asym<T,2>;
template<typename T> using _asym3 = _asym<T,3>;
template<typename T> using _asym4 = _asym<T,4>;

//forward-declare, body is below all tensor classes.

template<typename T>
requires is_tensor_v<T>
T elemMul(T const & a, T const & b);

template<typename A, typename B>
requires IsBinaryTensorOp<A, B>
typename A::Scalar inner(A const & a, B const & b);

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
