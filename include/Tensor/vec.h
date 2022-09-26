#pragma once

/*
NEW VERSION
- no more template metaprograms, instead constexprs
- no more helper structs, instead requires
- no more lower and upper
- modeled around glsl
- maybe extensions into matlab syntax (nested tensor implicit * contraction mul)

TODO:
	symmetric and antisymmetric matrices
	operator() reference drilling
	index notation still
*/

#include <tuple>
#include "Common/String.h"

namespace Tensor {
namespace v2 {

//for giving operators to the Cons and Prim vector classes
//how can you add correctly-typed ops via crtp to a union?
//unions can't inherit.
//until then...

#define TENSOR2_ADD_VECTOR_OP_EQ(classname, op)\
	classname& operator op(classname const & b) {\
		for (int i = 0; i < dim; ++i) {\
			s[i] op b.s[i];\
		}\
		return *this;\
	}

#define TENSOR2_ADD_SCALAR_OP_EQ(classname, op)\
	classname& operator op(ScalarType const & b) {\
		for (int i = 0; i < dim; ++i) {\
			s[i] op b;\
		}\
		return *this;\
	}

#define TENSOR2_ADD_CMP_OP(classname)\
	bool operator==(classname const & b) const {\
		for (int i = 0; i < dim; ++i) {\
			if (s[i] != b.s[i]) return false;\
		}\
		return true;\
	}\
	bool operator!=(classname const & b) const {\
		return !operator==(b);\
	}

// danger ... danger ...
#define TENSOR2_ADD_CAST_BOOL_OP()\
	operator bool() const {\
		for (int i = 0; i < dim; ++i) {\
			if (s[i] != T()) return true;\
		}\
		return false;\
	}

// danger ... danger ...
#define TENSOR2_ADD_CAST_OP(classname)\
	template<int dim2, typename U>\
	operator classname<U, dim2>() const {\
		classname<U, dim2> res;\
		for (int i = 0; i < dim && i < dim2; ++i) {\
			res.s[i] = (U)s[i];\
		}\
		return res;\
	}
	
#define TENSOR2_ADD_UNM(classname)\
	classname operator-() const {\
		classname result;\
		for (int i = 0; i < dim; ++i) {\
			result.s[i] = -s[i];\
		}\
		return result;\
	}

#define TENSOR2_ADD_DOT(classname)\
	T dot(classname const & b) const {\
		T result = {};\
		for (int i = 0; i < dim; ++i) {\
			result += s[i] * b.s[i];\
		}\
		return result;\
	}\
	T lenSq() const { return dot(*this); }\
	T length() const { return (T)sqrt(lenSq()); }

// danger ... danger ...
#define TENSOR2_ADD_CTOR_FROM_VEC(classname, othername)\
	template<typename U, int dim2>\
	classname(othername<U, dim2> const & v) {\
		int i = 0;\
		for (; i < dim && i < dim2; ++i) {\
			s[i] = (T)v[i];\
		}\
		for (; i < dim; ++i) {\
			s[i] = {};\
		}\
	}

#define TENSOR2_ADD_BRACKET_INDEX()\
	T & operator[](int i) { return s[i]; }\
	T const & operator[](int i) const { return s[i]; }

#define TENSOR2_ADD_OPS(classname)\
	ScalarType & operator()(int i) { return s[i]; }\
	ScalarType const & operator()(int i) const { return s[i]; }\
\
	TENSOR2_ADD_VECTOR_OP_EQ(classname, +=)\
	TENSOR2_ADD_VECTOR_OP_EQ(classname, -=)\
	TENSOR2_ADD_VECTOR_OP_EQ(classname, *=)\
	TENSOR2_ADD_VECTOR_OP_EQ(classname, /=)\
	TENSOR2_ADD_SCALAR_OP_EQ(classname, +=)\
	TENSOR2_ADD_SCALAR_OP_EQ(classname, -=)\
	TENSOR2_ADD_SCALAR_OP_EQ(classname, *=)\
	TENSOR2_ADD_SCALAR_OP_EQ(classname, /=)\
	TENSOR2_ADD_UNM(classname)\
	TENSOR2_ADD_DOT(classname)\
	TENSOR2_ADD_CMP_OP(classname)\
	TENSOR2_ADD_BRACKET_INDEX()
//	TENSOR2_ADD_CAST_BOOL_OP()

#if 0	//hmm, this isn't working when it is run
	classname& operator=(classname const & o) {\
		for (int i = 0; i < dim; ++i) {\
			s[i] = o.s[i];\
		}\
		return *this;\
	}
#endif


//use the 'rank' field to check and see if we're in a _vec or not
// TODO use something more specific to _vec in case other non-_vec classes use 'rank'
template<typename T>
constexpr bool has_rank_v = requires(T const & t) { T::rank; };

// base case
template<typename T>
struct VectorTraits {
	static constexpr int rank = 0;
	using ScalarType = T;
};

// recursive case
template<typename T> requires has_rank_v<T> 
struct VectorTraits<T> {
	static constexpr int rank = T::rank;
	using ScalarType = typename T::ScalarType;
};


//default
template<typename T, int dim_>
struct _vec {
	using This = _vec;
	using ScalarType = T;				//for TENSOR2_ADD_OPS
	static constexpr int dim = dim_;	//for TENSOR2_ADD_OPS

	T s[dim];
	_vec() {}

	TENSOR2_ADD_OPS(_vec)
	//TENSOR2_ADD_CAST_OP(_vec)
};

// to help the template evaluations of matrices ... here's the 1D vector
template<typename T>
struct _vec<T,1> {
	using This = _vec;
	using InnerType = T;
	using ScalarType = typename VectorTraits<T>::ScalarType;
	static constexpr int dim = 1;
	static constexpr int rank = 1 + VectorTraits<T>::rank;

	union {
		struct {
			T x = {};
		};
		struct {
			T s0;
		};
		T s[dim];
	};
	_vec() {}
	_vec(T x_) : x(x_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x)
	);

	TENSOR2_ADD_OPS(_vec)
	//TENSOR2_ADD_CAST_OP(_vec)
};

template<typename T>
using _vec1 = _vec<T,1>;

using bool1 = _vec1<bool>;
using uchar1 = _vec1<unsigned char>;
using int1 = _vec1<int>;
using float1 = _vec1<float>;
using double1 = _vec1<double>;

static_assert(std::is_same_v<float1::ScalarType, float>);
static_assert(std::is_same_v<float1::InnerType, float>);
static_assert(float1::rank == 1);
static_assert(float1::dim == 1);


template<typename T>
struct _vec<T,2> {
	using This = _vec;
	using InnerType = T;
	using ScalarType = typename VectorTraits<T>::ScalarType;
	static constexpr int dim = 2;
	static constexpr int rank = 1 + VectorTraits<T>::rank;

	union {
		struct {
			T x = {};
			T y = {};
		};
		struct {
			T s0;
			T s1;
		};
		T s[dim];
	};
	_vec() {}
	_vec(T x_, T y_) : x(x_), y(y_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y)
	);

	TENSOR2_ADD_OPS(_vec)
	//TENSOR2_ADD_CAST_OP(_vec)
};

template<typename T>
using _vec2 = _vec<T,2>;

using bool2 = _vec2<bool>;
using uchar2 = _vec2<unsigned char>;
using int2 = _vec2<int>;
using uint2 = _vec2<unsigned int>;
using float2 = _vec2<float>;
using double2 = _vec2<double>;

static_assert(std::is_same_v<float2::ScalarType, float>);
static_assert(std::is_same_v<float2::InnerType, float>);
static_assert(float2::rank == 1);
static_assert(float2::dim == 2);


template<typename T>
struct _vec<T,3> {
	using This = _vec;
	using InnerType = T;
	using ScalarType = typename VectorTraits<T>::ScalarType;
	static constexpr int dim = 3;
	static constexpr int rank = 1 + VectorTraits<T>::rank;

	union {
		struct {
			T x = {};
			T y = {};
			T z = {};
		};
		struct {
			T s0;
			T s1;
			T s2;
		};
		T s[dim];
	};
	_vec() {}
	_vec(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y),
		std::make_pair("z", &This::z)
	);

	TENSOR2_ADD_OPS(_vec)
	//TENSOR2_ADD_CAST_OP(_vec)
};

template<typename T>
using _vec3 = _vec<T,3>;

using bool3 = _vec3<bool>;
using uchar3 = _vec3<unsigned char>;
using int3 = _vec3<int>;
using float3 = _vec3<float>;
using double3 = _vec3<double>;

static_assert(std::is_same_v<float3::ScalarType, float>);
static_assert(std::is_same_v<float3::InnerType, float>);
static_assert(float3::rank == 1);
static_assert(float3::dim == 3);


template<typename T>
struct _vec<T,4> {
	using This = _vec;
	using InnerType = T;
	using ScalarType = typename VectorTraits<T>::ScalarType;
	static constexpr int dim = 4;
	static constexpr int rank = 1 + VectorTraits<T>::rank;

	union {
		struct {
			T x = {};
			T y = {};
			T z = {};
			T w = {};
		};
		struct {
			T s0;
			T s1;
			T s2;
			T s3;
		};
		T s[dim];
	};
	_vec() {}
	_vec(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y),
		std::make_pair("z", &This::z),
		std::make_pair("w", &This::w)
	);

	TENSOR2_ADD_OPS(_vec)
	//TENSOR2_ADD_CAST_OP(_vec)
};

template<typename T>
using _vec4 = _vec<T,4>;

using bool4 = _vec4<bool>;
using uchar4 = _vec4<unsigned char>;
using int4 = _vec4<int>;
using uint4 = _vec4<unsigned int>;
using float4 = _vec4<float>;
using double4 = _vec4<double>;

static_assert(std::is_same_v<float4::ScalarType, float>);
static_assert(std::is_same_v<float4::InnerType, float>);
static_assert(float4::rank == 1);
static_assert(float4::dim == 4);

//convention?  row-major to match math indexing, easy C inline ctor,  so A_ij = A[i][j]
// ... but OpenGL getFloatv(GL_...MATRIX) uses column-major so uploads have to be transposed
// ... also GLSL is column-major so between this and GLSL the indexes have to be transposed.
template<typename T, int dim1, int dim2> using _mat = _vec<_vec<T, dim2>, dim1>;

template<typename T> using _mat2x2 = _vec2<_vec2<T>>;
using int2x2 = _mat2x2<int>;
using float2x2 = _mat2x2<float>;

template<typename T> using _mat2x3 = _vec2<_vec3<T>>;
using int2x3 = _mat2x3<int>;
using float2x3 = _mat2x3<float>;

template<typename T> using _mat2x4 = _vec2<_vec4<T>>;
using int2x4 = _mat2x4<int>;
using float2x4 = _mat2x4<float>;

template<typename T> using _mat3x2 = _vec3<_vec2<T>>;
using int3x2 = _mat3x2<int>;
using float3x2 = _mat3x2<float>;

template<typename T> using _mat3x3 = _vec3<_vec3<T>>;
using int3x3 = _mat3x3<int>;
using float3x3 = _mat3x3<float>;

template<typename T> using _mat3x4 = _vec3<_vec4<T>>;
using int3x4 = _mat3x4<int>;
using float3x4 = _mat3x4<float>;

template<typename T> using _mat4x2 = _vec4<_vec2<T>>;
using int4x2 = _mat4x2<int>;
using float4x2 = _mat4x2<float>;

template<typename T> using _mat4x3 = _vec4<_vec3<T>>;
using int4x3 = _mat4x3<int>;
using float4x3 = _mat4x3<float>;

template<typename T> using _mat4x4 = _vec4<_vec4<T>>;
using int4x4 = _mat4x4<int>;
using float4x4 = _mat4x4<float>;

static_assert(std::is_same_v<float4x4::ScalarType, float>);
static_assert(std::is_same_v<float4x4::InnerType, float4>);
static_assert(float4x4::rank == 2);
static_assert(float4x4::dim == 4);
static_assert(float4x4::InnerType::dim == 4);

// vector op vector, matrix op matrix, and tensor op tensor per-component operators

#define TENSOR2_ADD_VECTOR_VECTOR_OP(op)\
template<typename T, int dim>\
_vec<T,dim> operator op(_vec<T,dim> const & a, _vec<T,dim> const & b) {\
	_vec<T,dim> c;\
	for (int i = 0; i < dim; ++i) {\
		c[i] = a[i] op b[i];\
	}\
	return c;\
}

TENSOR2_ADD_VECTOR_VECTOR_OP(+)
TENSOR2_ADD_VECTOR_VECTOR_OP(-)
TENSOR2_ADD_VECTOR_VECTOR_OP(/)

// vector * vector
//TENSOR2_ADD_VECTOR_VECTOR_OP(*) will cause ambiguous evaluation of matrix/matrix mul
// so it has to be constrained to only T == _vec<T,dim>:ScalarType
// c_i := a_i * b_i
template<typename T, int dim>
requires std::is_same_v<typename _vec<T,dim>::ScalarType, T>
_vec<T,dim> operator*(_vec<T, dim> const & a, _vec<T,dim> const & b) {
	_vec<T,dim> c;
	for (int i = 0; i < dim; ++i) {
		c[i] = a[i] * b[i];
	}
	return c;
}


// vector op scalar, scalar op vector, matrix op scalar, scalar op matrix, tensor op scalar, scalar op tensor operations
// need to require that T == _vec<T,dim>::ScalarType otherwise this will spill into vector/matrix operations
// c_i := a_i * b
// c_i := a * b_i

#define TENSOR2_ADD_VECTOR_SCALAR_OP(op)\
template<typename T, int dim>\
requires std::is_same_v<typename _vec<T,dim>::ScalarType, T>\
_vec<T,dim> operator op(_vec<T,dim> const & a, T const & b) {\
	_vec<T,dim> c;\
	for (int i = 0; i < dim; ++i) {\
		c[i] = a[i] op b;\
	}\
	return c;\
}\
template<typename T, int dim>\
requires std::is_same_v<typename _vec<T,dim>::ScalarType, T>\
_vec<T,dim> operator op(T const & a, _vec<T,dim> const & b) {\
	_vec<T,dim> c;\
	for (int i = 0; i < dim; ++i) {\
		c[i] = a op b[i];\
	}\
	return c;\
}

TENSOR2_ADD_VECTOR_SCALAR_OP(+)
TENSOR2_ADD_VECTOR_SCALAR_OP(-)
TENSOR2_ADD_VECTOR_SCALAR_OP(*)
TENSOR2_ADD_VECTOR_SCALAR_OP(/)


// matrix * matrix operators
// c_ik := a_ij * b_jk
template<typename T, int dim1, int dim2, int dim3>
requires std::is_same_v<typename _mat<T,dim1,dim2>::ScalarType, T>
_mat<T,dim1,dim3> operator*(_mat<T,dim1,dim2> const & a, _mat<T,dim2,dim3> const & b) {
	_mat<T,dim1,dim3> c;
	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim3; ++j) {
			T sum = {};
			for (int k = 0; k < dim2; ++k) {
				sum += a[i][k] * b[k][j];
			}
			c[i][j] = sum;
		}
	}
	return c;
}

// (row-)vector * matrix operator
// c_j := a_i * b_ij

template<typename T, int dim1, int dim2>
requires std::is_same_v<typename _mat<T,dim1,dim2>::ScalarType, T>
_vec<T,dim2> operator*(_vec<T,dim1> const & a, _mat<T,dim1,dim2> const & b) {
	_vec<T,dim2> c;
	for (int j = 0; j < dim2; ++j) {
		T sum = {};
		for (int i = 0; i < dim1; ++i) {
			sum += a[i] * b[i][j];
		}
		c[j] = sum;
	}
	return c;
}

// matrix * (column-)vector operator
// c_i := a_ij * b_j

template<typename T, int dim1, int dim2>
requires std::is_same_v<typename _mat<T,dim1,dim2>::ScalarType, T>
_vec<T,dim1> operator*(_mat<T,dim1,dim2> const & a, _vec<T,dim2> const & b) {
	_vec<T,dim1> c;
	for (int i = 0; i < dim1; ++i) {
		T sum = {};
		for (int j = 0; j < dim2; ++j) {
			sum += a[i][j] * b[j];
		}
		c[i] = sum;
	}
	return c;
}


// TODO GENERALIZE TO TENSOR MULTIPLICATIONS
// c_i1...i{p}_j1_..._j{q} = Σ_k1...k{r} a_i1_..._i{p}_k1_...k{r} * b_k1_..._k{r}_j1_..._j{q}

//matrixCompMult = component-wise multiplication, GLSL name compat
// c_ij := a_ij * b_ij

template<typename T, int dim1, int dim2>
requires std::is_same_v<typename _mat<T,dim1,dim2>::ScalarType, T>
_mat<T,dim1,dim2> matrixCompMult(_mat<T,dim1,dim2> const & a, _mat<T,dim1,dim2> const & b) {
	_mat<T,dim1,dim2> c;
	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim2; ++j) {
			c[i][j] = a[i][j] * b[i][j];
		}
	}
	return c;
}

// vector functions

// c := a_i * b_i
// TODO generalize
template<typename T, int dim>
requires std::is_same_v<typename _vec<T,dim>::ScalarType, T>
T dot(_vec<T,dim> const & a, _vec<T,dim> const & b) {
	T sum = {};
	for (int i = 0; i < dim; ++i) {
		sum += a[i] * b[i];
	}
	return sum;
}

template<typename T, int dim>
T lenSq(_vec<T,dim> const & v) {
	return dot(v,v);
}

template<typename T, int dim>
T length(_vec<T,dim> const & v) {
	return sqrt(lenSq(v));
}

template<typename T, int dim>
T distance(_vec<T,dim> const & a, _vec<T,dim> const & b) {
	return length(b - a);
}

template<typename T, int dim>
T normalize(_vec<T,dim> const & v) {
	return v / length(v);
}

// c_i := ε_ijk * b_j * c_k
template<typename T>
requires std::is_same_v<typename _vec3<T>::ScalarType, T>
_vec3<T> cross(_vec3<T> const & a, _vec3<T> const & b) {
	return _vec3<T>(
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]);
}

// c_ij := a_i * b_j
// TODO generalize to tensors c_i1_..ip_j1_..jq = a_i1..ip * b_j1..jq
template<typename T, int dim1, int dim2>
requires std::is_same_v<typename _vec<T,dim1>::ScalarType, T>
_mat<T,dim1,dim2> outerProduct(_vec<T,dim1> const & a, _vec<T,dim2> const & b) {
	_mat<T,dim1,dim2> c;
	for (int i = 0; i < dim1; ++i) {
		for (int j = 0; j < dim2; ++j) {
			c[i][j] = a[i] * b[j];
		}
	}
	return c;
}

// matrix functions

template<typename T, int dim1, int dim2>
_mat<T,dim2,dim1> transpose(_mat<T,dim1,dim2> const & a) {
	_mat<T,dim2,dim1> at;
	for (int i = 0; i < dim2; ++i) {
		for (int j = 0; j < dim1; ++j) {
			at[i][j] = a[j][i];
		}
	}
	return at;
}

template<typename T, int dim>
T determinant(_mat<T,dim,dim> const & a);

template<typename T>
T determinant(_mat2x2<T> const & a) {
	return a.x.x * a.y.y - a.x.y * a.y.x;
}
template<typename T>
T determinant(_mat3x3<T> const & a) {
	return a.x.x * a.y.y * a.z.z
		+ a.y.x * a.z.y * a.x.z
		+ a.z.x * a.x.y * a.y.z
		- a.x.z * a.y.y * a.z.x
		- a.y.z * a.z.y * a.x.x
		- a.z.z * a.x.y * a.y.x;
}

template<typename T>
T determinant(_mat4x4<T> const & a) {
	//autogen'd with symmath
	T const tmp1 = a.s2.s2 * a.s3.s3;
	T const tmp2 = a.s2.s3 * a.s3.s2;
	T const tmp3 = a.s2.s1 * a.s3.s3;
	T const tmp4 = a.s2.s3 * a.s3.s1;
	T const tmp5 = a.s2.s1 * a.s3.s2;
	T const tmp6 = a.s2.s2 * a.s3.s1;
	T const tmp7 = a.s2.s0 * a.s3.s3;
	T const tmp8 = a.s2.s3 * a.s3.s0;
	T const tmp9 = a.s2.s0 * a.s3.s2;
	T const tmp10 = a.s2.s2 * a.s3.s0;
	T const tmp11 = a.s2.s0 * a.s3.s1;
	T const tmp12 = a.s2.s1 * a.s3.s0;
	return a.s0.s0 * a.s1.s1 * tmp1 
		- a.s0.s0 * a.s1.s1 * tmp2 
		- a.s0.s0 * a.s1.s2 * tmp3
		+ a.s0.s0 * a.s1.s2 * tmp4
		+ a.s0.s0 * a.s1.s3 * tmp5 
		- a.s0.s0 * a.s1.s3 * tmp6 
		- a.s0.s1 * a.s1.s0 * tmp1
		+ a.s0.s1 * a.s1.s0 * tmp2
		+ a.s0.s1 * a.s1.s2 * tmp7 
		- a.s0.s1 * a.s1.s2 * tmp8 
		- a.s0.s1 * a.s1.s3 * tmp9
		+ a.s0.s1 * a.s1.s3 * tmp10
		+ a.s0.s2 * a.s1.s0 * tmp3 
		- a.s0.s2 * a.s1.s0 * tmp4 
		- a.s0.s2 * a.s1.s1 * tmp7
		+ a.s0.s2 * a.s1.s1 * tmp8
		+ a.s0.s2 * a.s1.s3 * tmp11 
		- a.s0.s2 * a.s1.s3 * tmp12 
		- a.s0.s3 * a.s1.s0 * tmp5
		+ a.s0.s3 * a.s1.s0 * tmp6
		+ a.s0.s3 * a.s1.s1 * tmp9 
		- a.s0.s3 * a.s1.s1 * tmp10
		+ a.s0.s3 * a.s1.s2 * tmp12 
		- a.s0.s3 * a.s1.s2 * tmp11;
}

template<typename T, int dim>
_mat<T,dim,dim> inverse(_mat<T,dim,dim> const & a, T const det);

template<typename T>
_mat3x3<T> inverse(_mat3x3<T> const & a, T const det) {
	T const invdet = T(1) / det;
	return {
		{
			invdet * (-a.s1.s2 * a.s2.s1 + a.s1.s1 * a.s2.s2),
			invdet * (a.s0.s2 * a.s2.s1 + -a.s0.s1 * a.s2.s2),
			invdet * (-a.s0.s2 * a.s1.s1 + a.s0.s1 * a.s1.s2)
		}, {
			invdet * (a.s1.s2 * a.s2.s0 + -a.s1.s0 * a.s2.s2),
			invdet * (-a.s0.s2 * a.s2.s0 + a.s0.s0 * a.s2.s2),
			invdet * (a.s0.s2 * a.s1.s0 + -a.s0.s0 * a.s1.s2)
		}, {
			invdet * (-a.s1.s1 * a.s2.s0 + a.s1.s0 * a.s2.s1),
			invdet * (a.s0.s1 * a.s2.s0 + -a.s0.s0 * a.s2.s1),
			invdet * (-a.s0.s1 * a.s1.s0 + a.s0.s0 * a.s1.s1)
		}
	};
}

//from : https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
template<typename T>
_mat4x4<T> inverse(_mat4x4<T> const & a, T const det) {
	T const a2323 = a.s2.s2 * a.s3.s3 - a.s2.s3 * a.s3.s2;
	T const a1323 = a.s2.s1 * a.s3.s3 - a.s2.s3 * a.s3.s1;
	T const a1223 = a.s2.s1 * a.s3.s2 - a.s2.s2 * a.s3.s1;
	T const a0323 = a.s2.s0 * a.s3.s3 - a.s2.s3 * a.s3.s0;
	T const a0223 = a.s2.s0 * a.s3.s2 - a.s2.s2 * a.s3.s0;
	T const a0123 = a.s2.s0 * a.s3.s1 - a.s2.s1 * a.s3.s0;
	T const a2313 = a.s1.s2 * a.s3.s3 - a.s1.s3 * a.s3.s2;
	T const a1313 = a.s1.s1 * a.s3.s3 - a.s1.s3 * a.s3.s1;
	T const a1213 = a.s1.s1 * a.s3.s2 - a.s1.s2 * a.s3.s1;
	T const a2312 = a.s1.s2 * a.s2.s3 - a.s1.s3 * a.s2.s2;
	T const a1312 = a.s1.s1 * a.s2.s3 - a.s1.s3 * a.s2.s1;
	T const a1212 = a.s1.s1 * a.s2.s2 - a.s1.s2 * a.s2.s1;
	T const a0313 = a.s1.s0 * a.s3.s3 - a.s1.s3 * a.s3.s0;
	T const a0213 = a.s1.s0 * a.s3.s2 - a.s1.s2 * a.s3.s0;
	T const a0312 = a.s1.s0 * a.s2.s3 - a.s1.s3 * a.s2.s0;
	T const a0212 = a.s1.s0 * a.s2.s2 - a.s1.s2 * a.s2.s0;
	T const a0113 = a.s1.s0 * a.s3.s1 - a.s1.s1 * a.s3.s0;
	T const a0112 = a.s1.s0 * a.s2.s1 - a.s1.s1 * a.s2.s0;
	T const invdet = T(1) / det;
	return { 
		{
		   invdet *  (a.s1.s1 * a2323 - a.s1.s2 * a1323 + a.s1.s3 * a1223),
		   invdet * -(a.s0.s1 * a2323 - a.s0.s2 * a1323 + a.s0.s3 * a1223),
		   invdet *  (a.s0.s1 * a2313 - a.s0.s2 * a1313 + a.s0.s3 * a1213),
		   invdet * -(a.s0.s1 * a2312 - a.s0.s2 * a1312 + a.s0.s3 * a1212),
		}, {
		   invdet * -(a.s1.s0 * a2323 - a.s1.s2 * a0323 + a.s1.s3 * a0223),
		   invdet *  (a.s0.s0 * a2323 - a.s0.s2 * a0323 + a.s0.s3 * a0223),
		   invdet * -(a.s0.s0 * a2313 - a.s0.s2 * a0313 + a.s0.s3 * a0213),
		   invdet *  (a.s0.s0 * a2312 - a.s0.s2 * a0312 + a.s0.s3 * a0212),
		}, {
		   invdet *  (a.s1.s0 * a1323 - a.s1.s1 * a0323 + a.s1.s3 * a0123),
		   invdet * -(a.s0.s0 * a1323 - a.s0.s1 * a0323 + a.s0.s3 * a0123),
		   invdet *  (a.s0.s0 * a1313 - a.s0.s1 * a0313 + a.s0.s3 * a0113),
		   invdet * -(a.s0.s0 * a1312 - a.s0.s1 * a0312 + a.s0.s3 * a0112),
		}, {
		   invdet * -(a.s1.s0 * a1223 - a.s1.s1 * a0223 + a.s1.s2 * a0123),
		   invdet *  (a.s0.s0 * a1223 - a.s0.s1 * a0223 + a.s0.s2 * a0123),
		   invdet * -(a.s0.s0 * a1213 - a.s0.s1 * a0213 + a.s0.s2 * a0113),
		   invdet *  (a.s0.s0 * a1212 - a.s0.s1 * a0212 + a.s0.s2 * a0112),
		}
	};
}

template<typename T, int dim>
_mat<T,dim,dim> inverse(_mat<T,dim,dim> const & a) {
	return inverse(a, determinant(a));
}

template<typename T, int dim>
_mat<T,dim,dim> diagonalMatrix(_vec<T,dim> const & v) {
	_mat<T,dim,dim> a;
	for (int i = 0; i < dim; ++i) {
		a[i][i] = v[i];
	}
	return a;
}

} //v2
} //Tensor
