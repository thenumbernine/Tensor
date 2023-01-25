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
	// can't use vec<int,1> because it hasn't been declared yet
	//&& A::dims == vec<int,1>(3)
	//&& B::dims == vec<int,1>(3);
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
struct vec;

template<typename Inner, int localDim>
requires(localDim > 0)
struct zero;

template<typename Inner, int localDim>
requires (localDim > 0)
struct ident;

template<typename Inner, int localDim>
requires (localDim > 0)
struct sym;

template<typename Inner, int localDim>
requires (localDim > 0)
struct asym;

template<typename Inner, int localDim, int localRank>
requires(localDim > 0 && localRank > 2)
struct symR;

template<typename Inner, int localDim, int localRank>
requires(localDim > 0 && localRank > 2)
struct asymR;


// hmm, I'm trying to use these storage_*'s in combination with is_instance_v<T, storage_*<dim>::template type> but it's failing, so here they are specialized
template<typename T> struct is_vec : public std::false_type {};
template<typename T, int d> struct is_vec<vec<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_vec_v = is_vec<T>::value;

template<typename T> struct is_zero : public std::false_type {};
template<typename T, int d> struct is_zero<zero<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_zero_v = is_zero<T>::value;

template<typename T> struct is_ident : public std::false_type {};
template<typename T, int d> struct is_ident<ident<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_ident_v = is_ident<T>::value;

template<typename T> struct is_sym : public std::false_type {};
template<typename T, int d> struct is_sym<sym<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_sym_v = is_sym<T>::value;

template<typename T> struct is_asym : public std::false_type {};
template<typename T, int d> struct is_asym<asym<T,d>> : public std::true_type {};
template<typename T> constexpr bool is_asym_v = is_asym<T>::value;

template<typename T> struct is_symR : public std::false_type {};
template<typename T, int d, int r> struct is_symR<symR<T,d,r>> : public std::true_type {};
template<typename T> constexpr bool is_symR_v = is_symR<T>::value;

template<typename T> struct is_asymR : public std::false_type {};
template<typename T, int d, int r> struct is_asymR<asymR<T,d,r>> : public std::true_type {};
template<typename T> constexpr bool is_asymR_v = is_asymR<T>::value;

//convention?  row-major to match math indexing, easy C inline ctor,  so A_ij = A[i][j]
// ... but OpenGL getFloatv(GL_...MATRIX) uses column-major so uploads have to be transposed
// ... also GLSL is column-major so between this and GLSL the indexes have to be transposed.
template<typename T, int dim1, int dim2> using mat = vec<vec<T, dim2>, dim1>;


// specific-sized templates
template<typename T> using vec2 = vec<T,2>;
template<typename T> using vec3 = vec<T,3>;
template<typename T> using vec4 = vec<T,4>;
template<typename T> using mat2x2 = vec2<vec2<T>>;
template<typename T> using mat2x3 = vec2<vec3<T>>;
template<typename T> using mat2x4 = vec2<vec4<T>>;
template<typename T> using mat3x2 = vec3<vec2<T>>;
template<typename T> using mat3x3 = vec3<vec3<T>>;
template<typename T> using mat3x4 = vec3<vec4<T>>;
template<typename T> using mat4x2 = vec4<vec2<T>>;
template<typename T> using mat4x3 = vec4<vec3<T>>;
template<typename T> using mat4x4 = vec4<vec4<T>>;
template<typename T> using sym2 = sym<T,2>;
template<typename T> using sym3 = sym<T,3>;
template<typename T> using sym4 = sym<T,4>;
template<typename T> using asym2 = asym<T,2>;
template<typename T> using asym3 = asym<T,3>;
template<typename T> using asym4 = asym<T,4>;


// dense vec-of-vec

// some template metaprogram helpers
//  needed for the math function
//  including operators, esp *

// tensori helpers:
// tensor<Scalar, storage_vec<dim>, storage_vec<dim2>, ..., storage_vec<dimN>>
//  use storage_sym<> storage_asym<> for injecting storage optimization
// tensor<Scalar, storage_sym<dim1>, ..., dimN>

template<int dim>
struct storage_vec {
	template<typename Inner>
	using type = vec<Inner,dim>;
	// so (hopefully) storage_vec<dim><Inner> == vec<dim,Inner>
};

template<int dim>
struct storage_zero {
	template<typename Inner>
	using type = zero<Inner,dim>;
};

template<int dim>
struct storage_sym {
	template<typename Inner>
	using type = sym<Inner,dim>;
};

template<int dim>
struct storage_asym {
	template<typename Inner>
	using type = asym<Inner,dim>;
};

template<int dim>
struct storage_ident {
	template<typename Inner>
	using type = ident<Inner,dim>;
};

template<int dim, int rank>
struct storage_symR {
	template<typename Inner>
	using type = symR<Inner,dim,rank>;
};

template<int dim, int rank>
struct storage_asymR {
	template<typename Inner>
	using type = asymR<Inner,dim,rank>;
};


// can I shorthand this? what is the syntax?
// this has a template and not a type on the lhs so I think no?
//template<int dim> using vecI = storage_vec<dim>::type;
//template<int dim> using symI = storage_sym<dim>::type;
//template<int dim> using asymI = storage_asym<dim>::type;

// useful helper macros, same as above but with transposed order

// tensori:
// tensor which allows custom nested storage, such as symmetric indexes

template<typename Scalar, typename... Storage>
struct tensori_impl;
template<typename Scalar, typename Storage, typename... MoreStorage>
struct tensori_impl<Scalar, Storage, MoreStorage...> {
	using type = typename Storage::template type<typename tensori_impl<Scalar, MoreStorage...>::type>;
};
template<typename Scalar, typename Storage>
struct tensori_impl<Scalar, Storage> {
	using type = typename Storage::template type<Scalar>;
};
template<typename Scalar>
struct tensori_impl<Scalar> {
	using type = Scalar;
};
template<typename Scalar, typename... Storage>
using tensori = typename tensori_impl<Scalar, Storage...>::type;

// type of a tensor with specific rank and dimension (for all indexes)
// used by some vec members

template<typename Scalar, int dim, int rank>
struct tensorr_impl {
	using type = vec<typename tensorr_impl<Scalar, dim, rank-1>::type, dim>;
};
template<typename Scalar, int dim>
struct tensorr_impl<Scalar, dim, 0> {
	using type = Scalar;
};
template<typename Src, int dim, int rank>
using tensorr = typename tensorr_impl<Src, dim, rank>::type;

// this is a useful enough one

template<typename Scalar, typename StorageTuple>
using tensorScalarTuple = Common::tuple_apply_t<tensori, Common::tuple_cat_t<std::tuple<Scalar>, StorageTuple>>;

// make a tensor from a list of dimensions
// ex: tensor<Scalar, dim1, ..., dimN>
// fully expanded storage - no spatial optimizations
// TODO can I accept template args as int or Index?
// maybe vararg function return type and decltype()?

template<typename Scalar, int dim, int... dims>
struct tensor_impl {
	using type = vec<typename tensor_impl<Scalar, dims...>::type, dim>;
};
template<typename Scalar, int dim>
struct tensor_impl<Scalar, dim> {
	using type = vec<Scalar,dim>;
};
template<typename Scalar, int dim, int... dims>
using tensor = typename tensor_impl<Scalar, dim, dims...>::type;

// useful helper for tensor:

template<typename Scalar, typename Seq>
struct tensorScalarSeqImpl;
template<typename Scalar, typename I, I i1, I... is>
struct tensorScalarSeqImpl<Scalar, std::integer_sequence<I, i1, is...>> {
	using type = tensor<Scalar, i1, is...>;
};
template<typename Scalar, typename Seq>
using tensorScalarSeq = typename tensorScalarSeqImpl<Scalar, Seq>::type;

/*
ok maybe this is a bad idea ..
tensorx< type, dim1, dim2, ... , storage char, [storage args] >
 where storage args are:
  -'z', dim = rank-1 zero index
  -'i', dim = rank-2 identity index
  -'s', dim = rank-2 symmetric
  -'a', dim = rank-2 antisymmetric
  -'S', dim, rank = rank-N symmetric
  -'A', dim, rank = rank-N antisymmetric
TODO rename tensor into tensorx and rename tensorx to tensor  ... so tensorx is for float3x3 float4x4x4 float2x3x4 etc, hence the 'x'
*/

template<typename Scalar, int... Args>
struct tensorx_impl;
template<typename Scalar, int dim, int... Args>
struct tensorx_impl<Scalar, dim, Args...> {
	using type = vec<typename tensorx_impl<Scalar, Args...>::type, dim>;
};
template<typename Scalar, int dim, int... Args>
struct tensorx_impl<Scalar, -'z', dim, Args...> {
	using type = zero<typename tensorx_impl<Scalar, Args...>::type, dim>;
};
template<typename Scalar, int dim, int... Args>
struct tensorx_impl<Scalar, -'i', dim, Args...> {
	using type = ident<typename tensorx_impl<Scalar, Args...>::type, dim>;
};
template<typename Scalar, int dim, int... Args>
struct tensorx_impl<Scalar, -'s', dim, Args...> {
	using type = sym<typename tensorx_impl<Scalar, Args...>::type, dim>;
};
template<typename Scalar, int dim, int... Args>
struct tensorx_impl<Scalar, -'a', dim, Args...> {
	using type = asym<typename tensorx_impl<Scalar, Args...>::type, dim>;
};
template<typename Scalar, int dim, int rank, int... Args>
struct tensorx_impl<Scalar, -'S', dim, rank, Args...> {
	using type = symR<typename tensorx_impl<Scalar, Args...>::type, dim, rank>;
};
template<typename Scalar, int dim, int rank, int... Args>
struct tensorx_impl<Scalar, -'A', dim, rank, Args...> {
	using type = asymR<typename tensorx_impl<Scalar, Args...>::type, dim, rank>;
};
template<typename Scalar>
struct tensorx_impl<Scalar> {
	using type = Scalar;
};
template<typename Scalar, int... Args>
using tensorx = typename tensorx_impl<Scalar, Args...>::type;

}
