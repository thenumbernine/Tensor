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

template<typename Inner, int localDim>
requires (localDim > 0)
struct _vec;

template<typename Inner, int localDim>
requires (localDim > 0)
struct _ident;

template<typename Inner, int localDim>
requires (localDim > 0)
struct _sym;

template<typename Inner, int localDim>
requires (localDim > 0)
struct _asym;

template<typename Inner, int localDim, int localRank>
requires(localDim > 0 && localRank > 2)
struct _symR;

template<typename Inner, int localDim, int localRank>
requires(localDim > 0 && localRank > 2)
struct _asymR;


// hmm, I'm trying to use these index_*'s in combination with is_instance_v<T, index_*<dim>::template type> but it's failing, so here they are specialized
template<typename T> struct is_vec : public std::false_type {};
template<typename T, int d> struct is_vec<_vec<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_vec_v = is_vec<T>::value;

template<typename T> struct is_ident : public std::false_type {};
template<typename T, int d> struct is_ident<_ident<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_ident_v = is_ident<T>::value;

template<typename T> struct is_sym : public std::false_type {};
template<typename T, int d> struct is_sym<_sym<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_sym_v = is_sym<T>::value;

template<typename T> struct is_asym : public std::false_type {};
template<typename T, int d> struct is_asym<_asym<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_asym_v = is_asym<T>::value;

template<typename T> struct is_symR : public std::false_type {};
template<typename T, int d, int r> struct is_symR<_symR<T,d,r>> : public std::true_type {};
template<typename T> constexpr bool is_symR_v = is_symR<T>::value;

template<typename T> struct is_asymR : public std::false_type {};
template<typename T, int d, int r> struct is_asymR<_asymR<T,d,r>> : public std::true_type {};
template<typename T> constexpr bool is_asymR_v = is_asymR<T>::value;


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

}
