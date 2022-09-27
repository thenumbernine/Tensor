#pragma once

/*
NEW VERSION
- no more template metaprograms, instead constexprs
- no more helper structs, instead requires
- no more lower and upper
- modeled around glsl
- maybe extensions into matlab syntax (nested tensor implicit * contraction mul)
- math indexing.  A.i.j := a_ij, which means row-major storage (sorry OpenGL)

TODO:
	symmetric and antisymmetric matrices
	operator() reference drilling
	index notation still


alright conventions, esp with my _sym being in the mix ...
operator[i] will denote a single index.  this means for two-rank structs they will have to return an accessor object
operator(i) same
operator(i,j,k,...) should always be allowed for any number of index dereferencing ... which means it will need to be implemented on accessors as well ... and means it may return an accessor

for GLSL / CL convention, every struct should have when possible:
	.s[]  (except for reference types ofc ... but why am I even mixing them in?  cuz swizzle and std::tie)
direct access to the data ... should be done via .s[] for now at least

operators ... scalar / tensor, tensor/scalar, tensor/tensor 

the * operator for tensor/tensor should be an outer+contract, ex:
	rank-2 a * rank-2 b = a_ij * b_jk = rank-2 c_ik (matrix mul)
	rank-1 a * rank-1 b = rank-0 c (dot product) ... tho for GLSL compat this is a per-element vector
	rank-3 a * rank-3 b = a_ijk * b_klm = c_ijlm


*/

#include "Common/String.h"
#include <tuple>
#include <functional>	//reference_wrapper, also function<> is by Partial
#include <cmath>		//sqrt()

namespace Tensor {

#define TENSOR_VECTOR_HEADER(dim_)\
\
	/* TRUE FOR _vec (NOT FOR _sym) */\
	/* this is this particular dimension of our vector */\
	/* M = matrix<T,i,j> == vec<vec<T,j>,i> so M::dim == i and M::InnerType::dim == j  */\
	/*  i.e. M::dims = int2(i,j) and M::ith_dim<0> == i and M::ith_dim<1> == j */\
	static constexpr int dim = dim_;\
\
	/* TRUE FOR _vec (NOT FOR _sym) */\
	/*how much does this structure contribute to the overall rank. */\
	/* for _vec it is 1, for _sym it is 2 */\
	static constexpr int thisRank = 1;\
\
	/* TRUE FOR _vec (NOT FOR _sym) */\
	/*this is the storage size, used for iterting across 's' */\
	/* for vectors etc it is 's' */\
	/* for (anti?)symmetric it is N*(N+1)/2 */\
	static constexpr int count = dim;\

#define TENSOR_HEADER()\
\
	/*  TRUE FOR ALL TENSORS */\
	/*  this is the next most nested class, so vector-of-vector is a matrix. */\
	using InnerType = T;\
\
	/*  TRUE FOR ALL TENSORS */\
	/*  this is the child-most nested class that isn't in our math library. */\
	using ScalarType = typename VectorTraits<T>::ScalarType;\
\
	/*  TRUE FOR ALL TENSORS */\
	/*  this is the rank/degree/index/number of letter-indexes of your tensor. */\
	/*  for vectors-of-vectors this is the nesting. */\
	/*  if you use any (anti?)symmetric then those take up 2 ranks / 2 indexes each instead of 1-rank each. */\
	static constexpr int rank = thisRank + VectorTraits<T>::rank;


//for giving operators to the Cons and Prim vector classes
//how can you add correctly-typed ops via crtp to a union?
//unions can't inherit.
//until then...

#define TENSOR_ADD_VECTOR_OP_EQ(classname, op)\
	classname& operator op(classname const & b) {\
		for (int i = 0; i < count; ++i) {\
			s[i] op b.s[i];\
		}\
		return *this;\
	}

#define TENSOR_ADD_SCALAR_OP_EQ(classname, op)\
	classname& operator op(ScalarType const & b) {\
		for (int i = 0; i < count; ++i) {\
			s[i] op b;\
		}\
		return *this;\
	}

#define TENSOR_ADD_CMP_OP(classname)\
	bool operator==(classname const & b) const {\
		for (int i = 0; i < count; ++i) {\
			if (s[i] != b.s[i]) return false;\
		}\
		return true;\
	}\
	bool operator!=(classname const & b) const {\
		return !operator==(b);\
	}

//::dims returns the total nested dimensions as an int-vec
#define TENSOR_ADD_SIZE(classname)\
	template<int i> static constexpr int ith_dim = VectorTraits<This>::template calc_ith_dim<i>();\
	static constexpr auto dims() { return VectorTraits<This>::dims(); }

// danger ... danger ...
#define TENSOR_ADD_CAST_BOOL_OP()\
	operator bool() const {\
		for (int i = 0; i < count; ++i) {\
			if (s[i] != T()) return true;\
		}\
		return false;\
	}

// danger ... danger ...
#define TENSOR_ADD_CAST_OP(classname)\
	template<int dim2, typename U>\
	operator classname<U, dim2>() const {\
		classname<U, dim2> res;\
		for (int i = 0; i < count && i < dim2; ++i) {\
			res.s[i] = (U)s[i];\
		}\
		return res;\
	}
	
#define TENSOR_ADD_UNM(classname)\
	classname operator-() const {\
		classname result;\
		for (int i = 0; i < count; ++i) {\
			result.s[i] = -s[i];\
		}\
		return result;\
	}

#define TENSOR_ADD_DOT(classname)\
	T dot(classname const & b) const {\
		T result = {};\
		for (int i = 0; i < count; ++i) {\
			result += s[i] * b.s[i];\
		}\
		return result;\
	}\
	T lenSq() const { return dot(*this); }\
	T length() const { return (T)sqrt(lenSq()); }

// danger ... danger ...
#define TENSOR_ADD_CTOR_FROM_VEC(classname, othername)\
	template<typename U, int dim2>\
	classname(othername<U, dim2> const & v) {\
		int i = 0;\
		for (; i < count && i < dim2; ++i) {\
			s[i] = (T)v[i];\
		}\
		for (; i < count; ++i) {\
			s[i] = {};\
		}\
	}

// works for nesting _vec's, not for _sym's
// I am using operator[] as the de-facto correct reference
#define TENSOR_ADD_VECTOR_BRACKET_INDEX()\
	T & operator[](int i) { return s[i]; }\
	T const & operator[](int i) const { return s[i]; }

// operator() should default through operator[]
#define TENSOR_ADD_RECURSIVE_CALL_INDEX()\
\
	/* a(i1,i2,...) := a_i1_i2_... */\
	template<typename... Rest>\
	auto & operator()(int i, Rest... rest) {\
		return (*this)[i](rest...);\
	}\
\
	template<typename... Rest>\
	auto const & operator()(int i, Rest... rest) const {\
		return (*this)[i](rest...);\
	}

#define TENSOR_ADD_INT_VEC_CALL_INDEX()\
\
	/* a(intN(i,...)) */\
	template<int dim2>\
	auto & operator()(_vec<int,dim2> const & i) {\
		if constexpr (dim2 == 1) {\
			return (*this)[i(0)];\
		} else {\
			return (*this)[i(0)](i.template subset<dim2-1, 1>());\
		}\
	}\
	template<int dim2>\
	auto const & operator()(_vec<int,dim2> const & i) const {\
		if constexpr (dim2 == 1) {\
			return (*this)[i(0)];\
		} else {\
			return (*this)[i(0)](i.template subset<dim2-1, 1>());\
		}\
	}\
\
	/* same but for std::array */\
	/* NOTICE this is only because _vec::iterator needs a std::array for indexing */\
	/*  and that is because _vec<T> can't seem to use _vec<int> for indexes, gets an 'incomplete type' error */\
	template<int dim2>\
	auto & operator()(std::array<int,dim2> const & i) {\
		if constexpr (dim2 == 1) {\
			return (*this)[i[0]];\
		} else {\
			std::array<int,dim2-1> subi;\
			std::copy(i.begin() + 1, i.begin() + dim2, subi.begin());\
			return (*this)[i[0]](subi);\
		}\
	}\
	template<int dim2>\
	auto const & operator()(std::array<int,dim2> const & i) const {\
		if constexpr (dim2 == 1) {\
			return (*this)[i[0]];\
		} else {\
			std::array<int,dim2-1> subi;\
			std::copy(i.begin() + 1, i.begin() + dim2, subi.begin());\
			return (*this)[i[0]](subi);\
		}\
	}

#define TENSOR_ADD_CALL_INDEX()\
\
	/* a(i) := a_i */\
	auto & operator()(int i) { return (*this)[i]; }\
	auto const & operator()(int i) const { return (*this)[i]; }\
\
	TENSOR_ADD_RECURSIVE_CALL_INDEX()\
	TENSOR_ADD_INT_VEC_CALL_INDEX()

// danger ... danger ...
#define TENSOR_ADD_ASSIGN_OP(classname)\
	classname & operator=(classname const & o) {\
		for (int i = 0; i < count; ++i) {\
			s[i] = o.s[i];\
		}\
		return *this;\
	}

#define TENSOR_ADD_SUBSET_ACCESS()\
\
	/* assumes packed tensor */\
	template<int subdim, int offset>\
	_vec<T,subdim> & subset() {\
		static_assert(offset + subdim <= dim);\
		return *(_vec<T,subdim>*)(s+offset);\
	}\
\
	/* assumes packed tensor */\
	template<int subdim, int offset>\
	_vec<T,subdim> const & subset() const {\
		static_assert(offset + subdim <= dim);\
		return *(_vec<T,subdim>*)(s+offset);\
	}\
\
	/* assumes packed tensor */\
	template<int subdim>\
	_vec<T,subdim> & subset(int offset) {\
		return *(_vec<T,subdim>*)(s+offset);\
	}\
\
	/* assumes packed tensor */\
	template<int subdim>\
	_vec<T,subdim> const & subset(int offset) const {\
		return *(_vec<T,subdim>*)(s+offset);\
	}

//these are all per-element assignment operators, so they should work fine for vector- and for symmetric-
#define TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_VECTOR_OP_EQ(classname, +=)\
	TENSOR_ADD_VECTOR_OP_EQ(classname, -=)\
	TENSOR_ADD_VECTOR_OP_EQ(classname, *=)\
	TENSOR_ADD_VECTOR_OP_EQ(classname, /=)\
	TENSOR_ADD_SCALAR_OP_EQ(classname, +=)\
	TENSOR_ADD_SCALAR_OP_EQ(classname, -=)\
	TENSOR_ADD_SCALAR_OP_EQ(classname, *=)\
	TENSOR_ADD_SCALAR_OP_EQ(classname, /=)\
	TENSOR_ADD_UNM(classname)\
	TENSOR_ADD_DOT(classname)\
	TENSOR_ADD_CMP_OP(classname)\
	TENSOR_ADD_SIZE(classname)

#define TENSOR_VECTOR_CLASS_OPS(classname)\
	TENSOR_ADD_VECTOR_BRACKET_INDEX()\
	TENSOR_ADD_CALL_INDEX()\
	TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_SUBSET_ACCESS()\
\
	/* assumes InnerType operator* exists */\
	/* TODO name this 'product' since 'volume' is ambiguous cuz it could alos mean product-of-dims */\
	T volume() const {\
		T res = s[0];\
		for (int i = 1; i < count; ++i) {\
			res *= s[i];\
		}\
		return res;\
	}\
\
	/* iterators */\
	template<typename vec_constness>\
	struct ReadIterator {\
		using intN = _vec<int,rank>;\
		vec_constness & owner;\
		intN index;\
		ReadIterator(vec_constness & owner_, intN index_ = {}) : owner(owner_), index(index_) {}\
		ReadIterator(ReadIterator const & o) { operator=(o); }\
		ReadIterator(ReadIterator && o) { operator=(o); }\
		ReadIterator & operator=(ReadIterator const & o) {\
			owner = o.owner;\
			index = o.index;\
			return *this;\
		}\
		ReadIterator & operator=(ReadIterator && o) {\
			owner = o.owner;\
			index = o.index;\
			return *this;\
		}\
		bool operator==(ReadIterator const & o) const {\
			return &owner == &o.owner && index == o.index;\
		}\
		bool operator!=(ReadIterator const & o) const {\
			return !operator==(o);\
		}\
\
		template<int i>\
		struct Increment {\
			static bool exec(ReadIterator &iter) {\
				++iter.index[i];\
				if (iter.index[i] < vec_constness::template ith_dim<i>) return true;\
				if (i < rank-1) iter.index[i] = 0;\
				return false;\
			}\
		};\
\
		ReadIterator &operator++() {\
			Common::ForLoop<0,rank,Increment>::exec(*this);\
			return *this;\
		}\
		ReadIterator &operator++(int) {\
			Common::ForLoop<0,rank,Increment>::exec(*this);\
			return *this;\
		}\
\
		ScalarType & operator*() const {\
			if constexpr (rank == 1) {\
				return owner[index[0]];\
			} else {\
				return owner(index);\
			}\
		}\
\
		void to_ostream(std::ostream & o) const {\
			o << "ReadIterator(owner=" << &owner << ", index=" << index << ")";\
		}\
\
		static ReadIterator begin(vec_constness & v) {\
			return ReadIterator(v);\
		}\
		static ReadIterator end(vec_constness & v) {\
			intN index;\
			index[rank-1] = vec_constness::template ith_dim<rank-1>;\
			return ReadIterator(v, index);\
		}\
	};\
	using iterator = ReadIterator<This>;\
	iterator begin() { return iterator::begin(*this); }\
	iterator end() { return iterator::end(*this); }\
	using const_iterator = ReadIterator<This const>;\
	const_iterator begin() const { return const_iterator::begin(*this); }\
	const_iterator end() const { return const_iterator::end(*this); }\
	const_iterator cbegin() const { return const_iterator::begin(*this); }\
	const_iterator cend() const { return const_iterator::end(*this); }

#if 0	//hmm, this isn't working when it is run
	//TENSOR_ADD_ASSIGN_OP(classname)
#endif


/*
use the 'rank' field to check and see if we're in a _vec (or a _sym) or not
 TODO use something more specific to this file in case other classes elsewhere use 'rank'
*/
template<typename T>
constexpr bool is_tensor_v = requires(T const & t) { T::rank; };

template<typename T, int dim_>
struct _vec;

// base/scalar case
template<typename T>
struct VectorTraits {
	
	static constexpr int rank = 0;
	
	using ScalarType = T;
};

// recursive/vec/matrix/tensor case
template<typename T> requires is_tensor_v<T> 
struct VectorTraits<T> {
	
	static constexpr int rank = T::rank;
	
	using ScalarType = typename T::ScalarType;

	// a function for getting the i'th component of the size vector
	template<int i>
	static constexpr int calc_ith_dim() {
		static_assert(i >= 0, "you tried to get a negative size");
		static_assert(i < rank, "you tried to get an oob size");
		if constexpr (i < T::thisRank) {
			return T::dim;
		} else {
			return VectorTraits<typename T::InnerType>::template calc_ith_dim<i - T::thisRank>();
		}
	}

	// .. then for loop iterator in dims() function and just a single exceptional case in the dims() function is needed
	static constexpr auto dims() {
		// if this is a vector-of-scalars, such that the dims would be an int1, just use int
		if constexpr (T::rank == 1) {
			return T::dim;
		} else {
			// use an int[dim]
			_vec<int,rank> sizev;
#if 0
#error "TODO constexpr for loop.  I could use my template-based one, but that means one more inline'd class, which is ugly."
#else
			for (int i = 0; i < T::thisRank; ++i) {
				sizev.s[i] = T::dim;
			}
			if constexpr (T::thisRank < rank) {
				// special case reading from int
				if constexpr (T::InnerType::rank == 1) {
					sizev.s[T::thisRank] = VectorTraits<typename T::InnerType>::dims();
				} else {
					// assigning sub-vector
					sizev.template subset<rank-T::thisRank, T::thisRank>()
						= VectorTraits<typename T::InnerType>::dims();
				}
			}
#endif			
			return sizev;
		}
	}
};

// default

template<typename T, int dim_>
struct _vec {
	// this is this class.  useful for templates.  you'd be surprised.
	using This = _vec;

	TENSOR_VECTOR_HEADER(dim_)
	TENSOR_HEADER()

	T s[count] = {};
	_vec() {}
	
	TENSOR_VECTOR_CLASS_OPS(_vec)
	//TENSOR_ADD_CAST_OP(_vec)
};

// size == 2 specialization

template<typename T>
struct _vec<T,2> {
	using This = _vec;
	TENSOR_VECTOR_HEADER(2);
	TENSOR_HEADER()

	union {
		struct {
			T x = {};
			T y = {};
		};
		struct {
			T s0;
			T s1;
		};
		T s[count];
	};
	_vec() {}
	_vec(T x_, T y_) : x(x_), y(y_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y)
	);

	TENSOR_VECTOR_CLASS_OPS(_vec)
	//TENSOR_ADD_CAST_OP(_vec)

	// 2-component swizzles
#define TENSOR_VEC2_ADD_SWIZZLE2_ij(i, j)\
	auto i ## j () { return _vec<std::reference_wrapper<T>, 2>(i, j); }
#define TENSOR_VEC2_ADD_SWIZZLE2_i(i)\
	TENSOR_VEC2_ADD_SWIZZLE2_ij(i,x)\
	TENSOR_VEC2_ADD_SWIZZLE2_ij(i,y)
#define TENSOR3_VEC2_ADD_SWIZZLE2()\
	TENSOR_VEC2_ADD_SWIZZLE2_i(x)\
	TENSOR_VEC2_ADD_SWIZZLE2_i(y)
	TENSOR3_VEC2_ADD_SWIZZLE2()
	
	// 3-component swizzles
#define TENSOR_VEC2_ADD_SWIZZLE3_ijk(i, j, k)\
	auto i ## j ## k() { return _vec<std::reference_wrapper<T>, 3>(i, j, k); }
#define TENSOR_VEC2_ADD_SWIZZLE3_ij(i,j)\
	TENSOR_VEC2_ADD_SWIZZLE3_ijk(i,j,x)\
	TENSOR_VEC2_ADD_SWIZZLE3_ijk(i,j,y)
#define TENSOR_VEC2_ADD_SWIZZLE3_i(i)\
	TENSOR_VEC2_ADD_SWIZZLE3_ij(i,x)\
	TENSOR_VEC2_ADD_SWIZZLE3_ij(i,y)
#define TENSOR3_VEC2_ADD_SWIZZLE3()\
	TENSOR_VEC2_ADD_SWIZZLE3_i(x)\
	TENSOR_VEC2_ADD_SWIZZLE3_i(y)
	TENSOR3_VEC2_ADD_SWIZZLE3()

	// 4-component swizzles
#define TENSOR_VEC2_ADD_SWIZZLE4_ijkl(i, j, k, l)\
	auto i ## j ## k ## l() { return _vec<std::reference_wrapper<T>, 4>(i, j, k, l); }
#define TENSOR_VEC2_ADD_SWIZZLE4_ijk(i,j,k)\
	TENSOR_VEC2_ADD_SWIZZLE4_ijkl(i,j,k,x)\
	TENSOR_VEC2_ADD_SWIZZLE4_ijkl(i,j,k,y)
#define TENSOR_VEC2_ADD_SWIZZLE4_ij(i,j)\
	TENSOR_VEC2_ADD_SWIZZLE4_ijk(i,j,x)\
	TENSOR_VEC2_ADD_SWIZZLE4_ijk(i,j,y)
#define TENSOR_VEC2_ADD_SWIZZLE4_i(i)\
	TENSOR_VEC2_ADD_SWIZZLE4_ij(i,x)\
	TENSOR_VEC2_ADD_SWIZZLE4_ij(i,y)
#define TENSOR3_VEC2_ADD_SWIZZLE4()\
	TENSOR_VEC2_ADD_SWIZZLE4_i(x)\
	TENSOR_VEC2_ADD_SWIZZLE4_i(y)
	TENSOR3_VEC2_ADD_SWIZZLE4()

};

template<typename T>
using _vec2 = _vec<T,2>;

using bool2 = _vec2<bool>;
using uchar2 = _vec2<unsigned char>;
using int2 = _vec2<int>;
using uint2 = _vec2<unsigned int>;
using float2 = _vec2<float>;
using double2 = _vec2<double>;

static_assert(sizeof(bool2) == sizeof(bool) * 2);
static_assert(sizeof(uchar2) == sizeof(unsigned char) * 2);
static_assert(sizeof(int2) == sizeof(int) * 2);
static_assert(sizeof(uint2) == sizeof(unsigned int) * 2);
static_assert(sizeof(float2) == sizeof(float) * 2);
static_assert(sizeof(double2) == sizeof(double) * 2);
static_assert(std::is_same_v<float2::ScalarType, float>);
static_assert(std::is_same_v<float2::InnerType, float>);
static_assert(float2::rank == 1);
static_assert(float2::dim == 2);

// size == 3 specialization

template<typename T>
struct _vec<T,3> {
	using This = _vec;
	TENSOR_VECTOR_HEADER(3);
	TENSOR_HEADER()

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
		T s[count];
	};
	_vec() {}
	_vec(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y),
		std::make_pair("z", &This::z)
	);

	TENSOR_VECTOR_CLASS_OPS(_vec)
	//TENSOR_ADD_CAST_OP(_vec)

	// 2-component swizzles
#define TENSOR_VEC3_ADD_SWIZZLE2_ij(i, j)\
	auto i ## j () { return _vec<std::reference_wrapper<T>, 2>(i, j); }
#define TENSOR_VEC3_ADD_SWIZZLE2_i(i)\
	TENSOR_VEC3_ADD_SWIZZLE2_ij(i,x)\
	TENSOR_VEC3_ADD_SWIZZLE2_ij(i,y)\
	TENSOR_VEC3_ADD_SWIZZLE2_ij(i,z)
#define TENSOR3_VEC3_ADD_SWIZZLE2()\
	TENSOR_VEC3_ADD_SWIZZLE2_i(x)\
	TENSOR_VEC3_ADD_SWIZZLE2_i(y)\
	TENSOR_VEC3_ADD_SWIZZLE2_i(z)
	TENSOR3_VEC3_ADD_SWIZZLE2()
	
	// 3-component swizzles
#define TENSOR_VEC3_ADD_SWIZZLE3_ijk(i, j, k)\
	auto i ## j ## k() { return _vec<std::reference_wrapper<T>, 3>(i, j, k); }
#define TENSOR_VEC3_ADD_SWIZZLE3_ij(i,j)\
	TENSOR_VEC3_ADD_SWIZZLE3_ijk(i,j,x)\
	TENSOR_VEC3_ADD_SWIZZLE3_ijk(i,j,y)\
	TENSOR_VEC3_ADD_SWIZZLE3_ijk(i,j,z)
#define TENSOR_VEC3_ADD_SWIZZLE3_i(i)\
	TENSOR_VEC3_ADD_SWIZZLE3_ij(i,x)\
	TENSOR_VEC3_ADD_SWIZZLE3_ij(i,y)\
	TENSOR_VEC3_ADD_SWIZZLE3_ij(i,z)
#define TENSOR3_VEC3_ADD_SWIZZLE3()\
	TENSOR_VEC3_ADD_SWIZZLE3_i(x)\
	TENSOR_VEC3_ADD_SWIZZLE3_i(y)\
	TENSOR_VEC3_ADD_SWIZZLE3_i(z)
	TENSOR3_VEC3_ADD_SWIZZLE3()

	// 4-component swizzles
#define TENSOR_VEC3_ADD_SWIZZLE4_ijkl(i, j, k, l)\
	auto i ## j ## k ## l() { return _vec<std::reference_wrapper<T>, 4>(i, j, k, l); }
#define TENSOR_VEC3_ADD_SWIZZLE4_ijk(i,j,k)\
	TENSOR_VEC3_ADD_SWIZZLE4_ijkl(i,j,k,x)\
	TENSOR_VEC3_ADD_SWIZZLE4_ijkl(i,j,k,y)\
	TENSOR_VEC3_ADD_SWIZZLE4_ijkl(i,j,k,z)
#define TENSOR_VEC3_ADD_SWIZZLE4_ij(i,j)\
	TENSOR_VEC3_ADD_SWIZZLE4_ijk(i,j,x)\
	TENSOR_VEC3_ADD_SWIZZLE4_ijk(i,j,y)\
	TENSOR_VEC3_ADD_SWIZZLE4_ijk(i,j,z)
#define TENSOR_VEC3_ADD_SWIZZLE4_i(i)\
	TENSOR_VEC3_ADD_SWIZZLE4_ij(i,x)\
	TENSOR_VEC3_ADD_SWIZZLE4_ij(i,y)\
	TENSOR_VEC3_ADD_SWIZZLE4_ij(i,z)
#define TENSOR3_VEC3_ADD_SWIZZLE4()\
	TENSOR_VEC3_ADD_SWIZZLE4_i(x)\
	TENSOR_VEC3_ADD_SWIZZLE4_i(y)\
	TENSOR_VEC3_ADD_SWIZZLE4_i(z)
	TENSOR3_VEC3_ADD_SWIZZLE4()
};

// TODO specialization for reference types -- don't initialize the s[] array (cuz in C++ you can't)

template<typename T>
using _vec3 = _vec<T,3>;

using bool3 = _vec3<bool>;
using uchar3 = _vec3<unsigned char>;
using int3 = _vec3<int>;
using uint3 = _vec3<unsigned int>;
using float3 = _vec3<float>;
using double3 = _vec3<double>;

static_assert(sizeof(bool3) == sizeof(bool) * 3);
static_assert(sizeof(uchar3) == sizeof(unsigned char) * 3);
static_assert(sizeof(int3) == sizeof(int) * 3);
static_assert(sizeof(uint3) == sizeof(unsigned int) * 3);
static_assert(sizeof(float3) == sizeof(float) * 3);
static_assert(sizeof(double3) == sizeof(double) * 3);
static_assert(std::is_same_v<float3::ScalarType, float>);
static_assert(std::is_same_v<float3::InnerType, float>);
static_assert(float3::rank == 1);
static_assert(float3::dim == 3);

template<typename T>
struct _vec<T,4> {
	using This = _vec;
	TENSOR_VECTOR_HEADER(4);
	TENSOR_HEADER()

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
		T s[count];
	};
	_vec() {}
	_vec(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y),
		std::make_pair("z", &This::z),
		std::make_pair("w", &This::w)
	);

	TENSOR_VECTOR_CLASS_OPS(_vec)
	//TENSOR_ADD_CAST_OP(_vec)

	// 2-component swizzles
#define TENSOR_VEC4_ADD_SWIZZLE2_ij(i, j)\
	auto i ## j () { return _vec<std::reference_wrapper<T>, 2>(i, j); }
#define TENSOR_VEC4_ADD_SWIZZLE2_i(i)\
	TENSOR_VEC4_ADD_SWIZZLE2_ij(i,x)\
	TENSOR_VEC4_ADD_SWIZZLE2_ij(i,y)\
	TENSOR_VEC4_ADD_SWIZZLE2_ij(i,z)\
	TENSOR_VEC4_ADD_SWIZZLE2_ij(i,w)
#define TENSOR3_VEC4_ADD_SWIZZLE2()\
	TENSOR_VEC4_ADD_SWIZZLE2_i(x)\
	TENSOR_VEC4_ADD_SWIZZLE2_i(y)\
	TENSOR_VEC4_ADD_SWIZZLE2_i(z)\
	TENSOR_VEC4_ADD_SWIZZLE2_i(w)
	TENSOR3_VEC4_ADD_SWIZZLE2()
	
	// 3-component swizzles
#define TENSOR_VEC4_ADD_SWIZZLE3_ijk(i, j, k)\
	auto i ## j ## k() { return _vec<std::reference_wrapper<T>, 3>(i, j, k); }
#define TENSOR_VEC4_ADD_SWIZZLE3_ij(i,j)\
	TENSOR_VEC4_ADD_SWIZZLE3_ijk(i,j,x)\
	TENSOR_VEC4_ADD_SWIZZLE3_ijk(i,j,y)\
	TENSOR_VEC4_ADD_SWIZZLE3_ijk(i,j,z)\
	TENSOR_VEC4_ADD_SWIZZLE3_ijk(i,j,w)
#define TENSOR_VEC4_ADD_SWIZZLE3_i(i)\
	TENSOR_VEC4_ADD_SWIZZLE3_ij(i,x)\
	TENSOR_VEC4_ADD_SWIZZLE3_ij(i,y)\
	TENSOR_VEC4_ADD_SWIZZLE3_ij(i,z)\
	TENSOR_VEC4_ADD_SWIZZLE3_ij(i,w)
#define TENSOR3_VEC4_ADD_SWIZZLE3()\
	TENSOR_VEC4_ADD_SWIZZLE3_i(x)\
	TENSOR_VEC4_ADD_SWIZZLE3_i(y)\
	TENSOR_VEC4_ADD_SWIZZLE3_i(z)\
	TENSOR_VEC4_ADD_SWIZZLE3_i(w)
	TENSOR3_VEC4_ADD_SWIZZLE3()

	// 4-component swizzles
#define TENSOR_VEC4_ADD_SWIZZLE4_ijkl(i, j, k, l)\
	auto i ## j ## k ## l() { return _vec<std::reference_wrapper<T>, 4>(i, j, k, l); }
#define TENSOR_VEC4_ADD_SWIZZLE4_ijk(i,j,k)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijkl(i,j,k,x)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijkl(i,j,k,y)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijkl(i,j,k,z)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijkl(i,j,k,w)
#define TENSOR_VEC4_ADD_SWIZZLE4_ij(i,j)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijk(i,j,x)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijk(i,j,y)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijk(i,j,z)\
	TENSOR_VEC4_ADD_SWIZZLE4_ijk(i,j,w)
#define TENSOR_VEC4_ADD_SWIZZLE4_i(i)\
	TENSOR_VEC4_ADD_SWIZZLE4_ij(i,x)\
	TENSOR_VEC4_ADD_SWIZZLE4_ij(i,y)\
	TENSOR_VEC4_ADD_SWIZZLE4_ij(i,z)\
	TENSOR_VEC4_ADD_SWIZZLE4_ij(i,w)
#define TENSOR3_VEC4_ADD_SWIZZLE4()\
	TENSOR_VEC4_ADD_SWIZZLE4_i(x)\
	TENSOR_VEC4_ADD_SWIZZLE4_i(y)\
	TENSOR_VEC4_ADD_SWIZZLE4_i(z)\
	TENSOR_VEC4_ADD_SWIZZLE4_i(w)
	TENSOR3_VEC4_ADD_SWIZZLE4()
};

template<typename T>
using _vec4 = _vec<T,4>;

using bool4 = _vec4<bool>;
using uchar4 = _vec4<unsigned char>;
using int4 = _vec4<int>;
using uint4 = _vec4<unsigned int>;
using float4 = _vec4<float>;
using double4 = _vec4<double>;

static_assert(sizeof(bool4) == sizeof(bool) * 4);
static_assert(sizeof(uchar4) == sizeof(unsigned char) * 4);
static_assert(sizeof(int4) == sizeof(int) * 4);
static_assert(sizeof(uint4) == sizeof(unsigned int) * 4);
static_assert(sizeof(float4) == sizeof(float) * 4);
static_assert(sizeof(double4) == sizeof(double) * 4);
static_assert(std::is_same_v<float4::ScalarType, float>);
static_assert(std::is_same_v<float4::InnerType, float>);
static_assert(float4::rank == 1);
static_assert(float4::dim == 4);


template<int dim> using boolN = _vec<bool, dim>;
template<int dim> using ucharN = _vec<unsigned char, dim>;
template<int dim> using intN = _vec<int, dim>;
template<int dim> using uintN = _vec<unsigned int, dim>;
template<int dim> using floatN = _vec<float, dim>;
template<int dim> using doubleN = _vec<double, dim>;


//convention?  row-major to match math indexing, easy C inline ctor,  so A_ij = A[i][j]
// ... but OpenGL getFloatv(GL_...MATRIX) uses column-major so uploads have to be transposed
// ... also GLSL is column-major so between this and GLSL the indexes have to be transposed.
template<typename T, int dim1, int dim2> using _mat = _vec<_vec<T, dim2>, dim1>;

template<typename T> using _mat2x2 = _vec2<_vec2<T>>;
using bool2x2 = _mat2x2<bool>;
using uchar2x2 = _mat2x2<unsigned char>;
using int2x2 = _mat2x2<int>;
using uint2x2 = _mat2x2<uint>;
using float2x2 = _mat2x2<float>;
using double2x2 = _mat2x2<double>;
static_assert(sizeof(bool2x2) == sizeof(bool) * 2 * 2);
static_assert(sizeof(uchar2x2) == sizeof(unsigned char) * 2 * 2);
static_assert(sizeof(int2x2) == sizeof(int) * 2 * 2);
static_assert(sizeof(uint2x2) == sizeof(unsigned int) * 2 * 2);
static_assert(sizeof(float2x2) == sizeof(float) * 2 * 2);
static_assert(sizeof(double2x2) == sizeof(double) * 2 * 2);

template<typename T> using _mat2x3 = _vec2<_vec3<T>>;
using bool2x3 = _mat2x3<bool>;
using uchar2x3 = _mat2x3<unsigned char>;
using int2x3 = _mat2x3<int>;
using uint2x3 = _mat2x3<uint>;
using float2x3 = _mat2x3<float>;
using double2x3 = _mat2x3<double>;
static_assert(sizeof(bool2x3) == sizeof(bool) * 2 * 3);
static_assert(sizeof(uchar2x3) == sizeof(unsigned char) * 2 * 3);
static_assert(sizeof(int2x3) == sizeof(int) * 2 * 3);
static_assert(sizeof(uint2x3) == sizeof(unsigned int) * 2 * 3);
static_assert(sizeof(float2x3) == sizeof(float) * 2 * 3);
static_assert(sizeof(double2x3) == sizeof(double) * 2 * 3);

template<typename T> using _mat2x4 = _vec2<_vec4<T>>;
using bool2x4 = _mat2x4<bool>;
using uchar2x4 = _mat2x4<unsigned char>;
using int2x4 = _mat2x4<int>;
using uint2x4 = _mat2x4<uint>;
using float2x4 = _mat2x4<float>;
using double2x4 = _mat2x4<double>;

template<typename T> using _mat3x2 = _vec3<_vec2<T>>;
using bool3x2 = _mat3x2<bool>;
using uchar3x2 = _mat3x2<unsigned char>;
using int3x2 = _mat3x2<int>;
using uint3x2 = _mat3x2<uint>;
using float3x2 = _mat3x2<float>;
using double3x2 = _mat3x2<double>;

template<typename T> using _mat3x3 = _vec3<_vec3<T>>;
using bool3x3 = _mat3x3<bool>;
using uchar3x3 = _mat3x3<unsigned char>;
using int3x3 = _mat3x3<int>;
using uint3x3 = _mat3x3<uint>;
using float3x3 = _mat3x3<float>;
using double3x3 = _mat3x3<double>;

template<typename T> using _mat3x4 = _vec3<_vec4<T>>;
using bool3x4 = _mat3x4<bool>;
using uchar3x4 = _mat3x4<unsigned char>;
using int3x4 = _mat3x4<int>;
using uint3x4 = _mat3x4<uint>;
using float3x4 = _mat3x4<float>;
using double3x4 = _mat3x4<double>;

template<typename T> using _mat4x2 = _vec4<_vec2<T>>;
using bool4x2 = _mat4x2<bool>;
using uchar4x2 = _mat4x2<unsigned char>;
using int4x2 = _mat4x2<int>;
using uint4x2 = _mat4x2<uint>;
using float4x2 = _mat4x2<float>;
using double4x2 = _mat4x2<double>;

template<typename T> using _mat4x3 = _vec4<_vec3<T>>;
using bool4x3 = _mat4x3<bool>;
using uchar4x3 = _mat4x3<unsigned char>;
using int4x3 = _mat4x3<int>;
using uint4x3 = _mat4x3<uint>;
using float4x3 = _mat4x3<float>;
using double4x3 = _mat4x3<double>;

template<typename T> using _mat4x4 = _vec4<_vec4<T>>;
using bool4x4 = _mat4x4<bool>;
using uchar4x4 = _mat4x4<unsigned char>;
using int4x4 = _mat4x4<int>;
using uint4x4 = _mat4x4<uint>;
using float4x4 = _mat4x4<float>;
using double4x4 = _mat4x4<double>;

static_assert(std::is_same_v<float4x4::ScalarType, float>);
static_assert(std::is_same_v<float4x4::InnerType, float4>);
static_assert(float4x4::rank == 2);
static_assert(float4x4::dim == 4);
static_assert(float4x4::InnerType::dim == 4);

// vector op vector, matrix op matrix, and tensor op tensor per-component operators

#define TENSOR_ADD_VECTOR_VECTOR_OP(op)\
template<typename T, int dim>\
_vec<T,dim> operator op(_vec<T,dim> const & a, _vec<T,dim> const & b) {\
	_vec<T,dim> c;\
	for (int i = 0; i < dim; ++i) {\
		c[i] = a[i] op b[i];\
	}\
	return c;\
}

TENSOR_ADD_VECTOR_VECTOR_OP(+)
TENSOR_ADD_VECTOR_VECTOR_OP(-)
TENSOR_ADD_VECTOR_VECTOR_OP(/)

// vector * vector
//TENSOR_ADD_VECTOR_VECTOR_OP(*) will cause ambiguous evaluation of matrix/matrix mul
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

#define TENSOR_ADD_VECTOR_SCALAR_OP(op)\
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

TENSOR_ADD_VECTOR_SCALAR_OP(+)
TENSOR_ADD_VECTOR_SCALAR_OP(-)
TENSOR_ADD_VECTOR_SCALAR_OP(*)
TENSOR_ADD_VECTOR_SCALAR_OP(/)


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
// TODO generalize to any sort of tensor swizzle

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
_mat<T,dim,dim> diagonalMatrix(_vec<T,dim> const & v) {
	_mat<T,dim,dim> a;
	for (int i = 0; i < dim; ++i) {
		a[i][i] = v[i];
	}
	return a;
}

// symmetric matrices

#define TENSOR_SYMMETRIC_MATRIX_HEADER(dim_)\
	static constexpr int dim = dim_;\
	static constexpr int thisRank = 2;\
	static constexpr int count = (dim * (dim + 1)) >> 1;

template<int dim>
int symIndex(int i, int j) {
	if (j > i) return symIndex<dim>(j,i);
	return ((i * (i + 1)) >> 1) + j;
}

//also symmetric has to define 1-arg operator()
// that means I can't use the default so i have to make a 2-arg recursive case
#define TENSOR_SYMMETRIC_ADD_RECURSIVE_CALL_INDEX()\
\
	/* a(i1,i2,...) := a_i1_i2_... */\
	template<typename... Rest>\
	auto & operator()(int i, int j, Rest... rest) {\
		return (*this)(i,j)(rest...);\
	}\
\
	template<typename... Rest>\
	auto const & operator()(int i, int j, Rest... rest) const {\
		return (*this)(i,j)(rest...);\
	}


/*
for the call index operator
a 1-param is incomplete, so it should return an accessor (same as operator[])
but a 2-param is complete
and a more-than-2 will return call on the []
and therein risks applying a call to an accessor
so the accessors need nested call indexing too
*/
#define TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(classname)\
	TENSOR_ADD_OPS(classname)\
\
	/* a(i,j) := a_ij = a_ji */\
	/* this is the direct acces */\
	InnerType & operator()(int i, int j) { return s[symIndex<dim>(i,j)]; }\
	InnerType const & operator()(int i, int j) const { return s[symIndex<dim>(i,j)]; }\
\
	struct Accessor {\
		classname & owner;\
		int i;\
		Accessor(classname & owner_, int i_) : owner(owner_), i(i_) {}\
		InnerType & operator[](int j) { return owner(i,j); }\
		TENSOR_ADD_CALL_INDEX()\
	};\
	Accessor operator[](int i) { return Accessor(*this, i); }\
\
	struct ConstAccessor {\
		classname const & owner;\
		int i;\
		ConstAccessor(classname const & owner_, int i_) : owner(owner_), i(i_) {}\
		InnerType const & operator[](int j) { return owner(i,j); }\
		TENSOR_ADD_CALL_INDEX()\
	};\
	ConstAccessor operator[](int i) const { return ConstAccessor(*this, i); }\
\
	/* a(i) := a_i */\
	/* this is incomplete so it returns the operator[] which returns the accessor */\
	Accessor & operator()(int i) { return (*this)[i]; }\
	ConstAccessor & operator()(int i) const { return (*this)[i]; }\
\
	TENSOR_ADD_INT_VEC_CALL_INDEX()\
	TENSOR_SYMMETRIC_ADD_RECURSIVE_CALL_INDEX()


template<typename T, int dim_>
struct _sym {
	using This = _sym;
	TENSOR_SYMMETRIC_MATRIX_HEADER(dim_)
	TENSOR_HEADER()

	T s[count] = {};
	_sym() {}

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(_sym)
	//TENSOR_ADD_CAST_OP(_sym)
};

template<typename T>
struct _sym<T,2> {
	using This = _sym;
	TENSOR_SYMMETRIC_MATRIX_HEADER(2)
	TENSOR_HEADER()

	union {
		struct {
			T xx = {};
			T xy = {};
			T yy = {};
		};
		struct {
			T s00;
			T s01;
			T s11;
		};
		T s[count];
	};
	_sym() {}
	_sym(T xx_, T xy_, T yy_) : xx(xx_), xy(xy_), yy(yy_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("xx", &This::xx),
		std::make_pair("xy", &This::xy),
		std::make_pair("yy", &This::yy)
	);

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(_sym)
	//TENSOR_ADD_CAST_OP(_sym)
};

template<typename T>
struct _sym<T,3> {
	using This = _sym;
	TENSOR_SYMMETRIC_MATRIX_HEADER(3)
	TENSOR_HEADER()

	union {
		struct {
			T xx = {};
			T xy = {};
			T yy = {};
			T xz = {};
			T yz = {};
			T zz = {};
		};
		struct {
			T s00;
			T s01;
			T s11;
			T s02;
			T s12;
			T s22;
		};
		T s[count];
	};
	_sym() {}
	_sym(
		T const & xx_,
		T const & xy_,
		T const & yy_,
		T const & xz_,
		T const & yz_,
		T const & zz_
	) : xx(xx_),
		xy(xy_),
		yy(yy_),
		xz(xz_),
		yz(yz_),
		zz(zz_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("xx", &This::xx),
		std::make_pair("xy", &This::xy),
		std::make_pair("yy", &This::yy),
		std::make_pair("xz", &This::xz),
		std::make_pair("yz", &This::yz),
		std::make_pair("zz", &This::zz)
	);

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(_sym)
	//TENSOR_ADD_CAST_OP(_sym)
};

template<typename T>
struct _sym<T,4> {
	using This = _sym;
	TENSOR_SYMMETRIC_MATRIX_HEADER(4)
	TENSOR_HEADER()

	union {
		struct {
			T xx = {};
			T xy = {};
			T yy = {};
			T xz = {};
			T yz = {};
			T zz = {};
			T xw = {};
			T yw = {};
			T zw = {};
			T ww = {};
		};
		struct {
			T s00;
			T s01;
			T s11;
			T s02;
			T s12;
			T s22;
			T s03;
			T s13;
			T s23;
			T s33;	
		};
		T s[count];
	};
	_sym() {}
	_sym(
		T const & xx_,
		T const & xy_,
		T const & yy_,
		T const & xz_,
		T const & yz_,
		T const & zz_,
		T const & xw_,
		T const & yw_,
		T const & zw_,
		T const & ww_
	) : xx(xx_),
		xy(xy_),
		yy(yy_),
		xz(xz_),
		yz(yz_),
		zz(zz_),
		xw(xw_),
		yw(yw_),
		zw(zw_),
		ww(ww_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("xx", &This::xx),
		std::make_pair("xy", &This::xy),
		std::make_pair("yy", &This::yy),
		std::make_pair("xz", &This::xz),
		std::make_pair("yz", &This::yz),
		std::make_pair("zz", &This::zz),
		std::make_pair("xw", &This::xw),
		std::make_pair("yw", &This::yw),
		std::make_pair("zw", &This::zw),
		std::make_pair("ww", &This::ww)
	);

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(_sym)
	//TENSOR_ADD_CAST_OP(_sym)
};

// symmetric op symmetric

#define TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(op)\
template<typename T, int dim>\
_sym<T,dim> operator op(_sym<T,dim> const & a, _sym<T,dim> const & b) {\
	_sym<T,dim> c;\
	for (int i = 0; i < c.count; ++i) {\
		c.s[i] = a.s[i] op b.s[i];\
	}\
	return c;\
}

TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(+)
TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(-)
TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(/)

// symmetric op scalar, scalar op symmetric

#define TENSOR_ADD_SYMMETRIC_MATRIX_SCALAR_OP(op)\
template<typename T, int dim>\
requires std::is_same_v<typename _sym<T,dim>::ScalarType, T>\
_sym<T,dim> operator op(_sym<T,dim> const & a, T const & b) {\
	_sym<T,dim> c;\
	for (int i = 0; i < c.count; ++i) {\
		c.s[i] = a.s[i] op b;\
	}\
	return c;\
}\
template<typename T, int dim>\
requires std::is_same_v<typename _sym<T,dim>::ScalarType, T>\
_sym<T,dim> operator op(T const & a, _sym<T,dim> const & b) {\
	_sym<T,dim> c;\
	for (int i = 0; i < c.count; ++i) {\
		c.s[i] = a op b.s[i];\
	}\
	return c;\
}

TENSOR_ADD_SYMMETRIC_MATRIX_SCALAR_OP(+)
TENSOR_ADD_SYMMETRIC_MATRIX_SCALAR_OP(-)
TENSOR_ADD_SYMMETRIC_MATRIX_SCALAR_OP(*)
TENSOR_ADD_SYMMETRIC_MATRIX_SCALAR_OP(/)

// how to name symmetric-optimized storage?

// using float2s2 in place of float2x2 does gieve the hint, but might be hard to read
// another option is floatSym2
template<typename T> using _sym2 = _sym<T,2>;
using bool2s2 = _sym2<bool>;
using uchar2s2 = _sym2<unsigned char>;
using int2s2 = _sym2<int>;
using uint2s2 = _sym2<uint>;
using float2s2 = _sym2<float>;
using double2s2 = _sym2<double>;

static_assert(sizeof(bool2s2) == sizeof(bool) * 3);
static_assert(sizeof(uchar2s2) == sizeof(unsigned char) * 3);
static_assert(sizeof(int2s2) == sizeof(int) * 3);
static_assert(sizeof(uint2s2) == sizeof(unsigned int) * 3);
static_assert(sizeof(float2s2) == sizeof(float) * 3);
static_assert(sizeof(double2s2) == sizeof(double) * 3);
static_assert(std::is_same_v<typename float2s2::ScalarType, float>);

template<typename T> using _sym3 = _sym<T,3>;
using bool3s3 = _sym3<bool>;
using uchar3s3 = _sym3<unsigned char>;
using int3s3 = _sym3<int>;
using uint3s3 = _sym3<uint>;
using float3s3 = _sym3<float>;
using double3s3 = _sym3<double>;

static_assert(sizeof(bool3s3) == sizeof(bool) * 6);
static_assert(sizeof(uchar3s3) == sizeof(unsigned char) * 6);
static_assert(sizeof(int3s3) == sizeof(int) * 6);
static_assert(sizeof(uint3s3) == sizeof(unsigned int) * 6);
static_assert(sizeof(float3s3) == sizeof(float) * 6);
static_assert(sizeof(double3s3) == sizeof(double) * 6);
static_assert(std::is_same_v<typename float3s3::ScalarType, float>);

template<typename T> using _sym4 = _sym<T,4>;
using bool4s4 = _sym4<bool>;
using uchar4s4 = _sym4<unsigned char>;
using int4s4 = _sym4<int>;
using uint4s4 = _sym4<uint>;
using float4s4 = _sym4<float>;
using double4s4 = _sym4<double>;

static_assert(sizeof(bool4s4) == sizeof(bool) * 10);
static_assert(sizeof(uchar4s4) == sizeof(unsigned char) * 10);
static_assert(sizeof(int4s4) == sizeof(int) * 10);
static_assert(sizeof(uint4s4) == sizeof(unsigned int) * 10);
static_assert(sizeof(float4s4) == sizeof(float) * 10);
static_assert(sizeof(double4s4) == sizeof(double) * 10);
static_assert(std::is_same_v<typename float4s4::ScalarType, float>);

}

// ostream
// _vec does have .fields
// and I do have my default .fields ostream
// but here's a manual override anyways
// so that the .fields vec2 vec3 vec4 and the non-.fields other vecs all look the same
#if 1
template<typename Type, int dim>
std::ostream & operator<<(std::ostream & o, Tensor::_vec<Type,dim> const & t) {
	char const * sep = "";
	o << "{";
	for (int i = 0; i < t.count; ++i) {
		o << sep << t(i);
		sep = ", ";
	}
	o << "}";
	return o;
}

// TODO print as fields of .sym, or print as vector?
template<typename Type, int dim>
std::ostream & operator<<(std::ostream & o, Tensor::_sym<Type,dim> const & t) {
	char const * sep = "";
	o << "{";
	for (int i = 0; i < t.count; ++i) {
		o << sep << t.s[i];
		sep = ", ";
	}
	o << "}";
	return o;
}
#endif

// tostring / ostream

namespace std {

template<typename T, int n>
std::string to_string(Tensor::_vec<T, n> const & x) {
	return Common::objectStringFromOStream(x);
}

}
