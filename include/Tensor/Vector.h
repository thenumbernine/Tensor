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


TODO TODO
	if I do row-major then
		- C bracket ctor is in the same layout as the matrix
		- C notation matches matrix notation : A[i0][i1] = A_i0_i1
		- memory layout is transposed from nested index order: A_i0_i1 = A[i1 + i0 * size0]
	if I do column-major then C inline indexing is transposed
		- C bracket ctor is transposed from the layout of the matrix
		- matrix notation is transposed: A[j][i] == A_ij
		- memory layout matches nested index order: A_i0_i1 = A[i0 + size0 * i1]
		- OpenGL uses this too.

*/

#include "Common/String.h"
#include <tuple>
#include <functional>	//reference_wrapper, also function<> is by Partial
#include <cmath>		//sqrt()

namespace Tensor {

#define TENSOR_VECTOR_HEADER(localDim_)\
\
	/* TRUE FOR _vec (NOT FOR _sym) */\
	/* this is this particular dimension of our vector */\
	/* M = matrix<T,i,j> == vec<vec<T,j>,i> so M::localDim == i and M::Inner::localDim == j  */\
	/*  i.e. M::dims = int2(i,j) and M::dim<0> == i and M::dim<1> == j */\
	static constexpr int localDim = localDim_;\
\
	/* TRUE FOR _vec (NOT FOR _sym) */\
	/*how much does this structure contribute to the overall rank. */\
	/* for _vec it is 1, for _sym it is 2 */\
	static constexpr int localRank = 1;\
\
	/* TRUE FOR _vec (NOT FOR _sym) */\
	/*this is the storage size, used for iterting across 's' */\
	/* for vectors etc it is 's' */\
	/* for (anti?)symmetric it is N*(N+1)/2 */\
	/* TODO make a 'count<int nesting>', same as dim? */\
	static constexpr int localCount = localDim;

#define TENSOR_HEADER()\
\
	/*  TRUE FOR ALL TENSORS */\
	/*  this is the next most nested class, so vector-of-vector is a matrix. */\
	using Inner = T;\
\
	/* some helper stuff associated with our _vec */\
	using Traits = VectorTraits<This>;\
\
	/*  TRUE FOR ALL TENSORS */\
	/*  this is the child-most nested class that isn't in our math library. */\
	/* TODO why can't I use VectorTraits<This> here? */\
	using Scalar = typename VectorTraits<Inner>::Scalar;\
\
	/*  TRUE FOR ALL TENSORS */\
	/*  this is the rank/degree/index/number of letter-indexes of your tensor. */\
	/*  for vectors-of-vectors this is the nesting. */\
	/*  if you use any (anti?)symmetric then those take up 2 ranks / 2 indexes each instead of 1-rank each. */\
	/**/\
	/* if we do recursive case in VectorTraits (failing for rank-3 and above): */\
	/*static constexpr int rank = VectorTraits<This>::rank;*/\
	/* if we do recursive case here: */\
	static constexpr int rank = localRank + VectorTraits<Inner>::rank;\
\
	/* used for vector-dereferencing into the tensor. */\
	using intN = _vec<int,rank>;\
\
	/* how many _vec<_vec<...'s there are ... no extra +'s for multi-rank nestings */\
	/* so _sym<> has numNestings=1 and rank=2, _vec<> has numNestings=1 and rank=1, _vec_vec<>>==_mat<> has numNestings=2 and rank=2 */\
	static constexpr int numNestings = Traits::numNestings;


//for giving operators to the Cons and Prim vector classes
//how can you add correctly-typed ops via crtp to a union?
//unions can't inherit.
//until then...

#define TENSOR_ADD_VECTOR_OP_EQ(classname, op)\
	classname& operator op(classname const & b) {\
		for (int i = 0; i < localCount; ++i) {\
			s[i] op b.s[i];\
		}\
		return *this;\
	}

#define TENSOR_ADD_SCALAR_OP_EQ(classname, op)\
	classname& operator op(Scalar const & b) {\
		for (int i = 0; i < localCount; ++i) {\
			s[i] op b;\
		}\
		return *this;\
	}

#define TENSOR_ADD_CMP_OP(classname)\
	bool operator==(classname const & b) const {\
		for (int i = 0; i < localCount; ++i) {\
			if (s[i] != b.s[i]) return false;\
		}\
		return true;\
	}\
	bool operator!=(classname const & b) const {\
		return !operator==(b);\
	}

//::dims returns the total nested dimensions as an int-vec
#define TENSOR_ADD_SIZE(classname)\
\
	template<int i> static constexpr int dim = Traits::template calc_ith_dim<i>();\
\
	static constexpr auto dims() { return Traits::dims(); }\
\
	template<int i> static constexpr int count = Traits::template Nested<i>::localCount;

// danger ... danger ...
#define TENSOR_ADD_CAST_BOOL_OP()\
	operator bool() const {\
		for (int i = 0; i < localCount; ++i) {\
			if (s[i] != T()) return true;\
		}\
		return false;\
	}

// danger ... danger ...
#define TENSOR_ADD_ASSIGN_OP(classname)\
	classname & operator=(classname const & o) {\
		for (int i = 0; i < localCount; ++i) {\
			s[i] = o.s[i];\
		}\
		return *this;\
	}

#define TENSOR_ADD_UNM(classname)\
	classname operator-() const {\
		classname result;\
		for (int i = 0; i < localCount; ++i) {\
			result.s[i] = -s[i];\
		}\
		return result;\
	}

#define TENSOR_ADD_DOT(classname)\
	T dot(classname const & b) const {\
		T result = {};\
		for (int i = 0; i < localCount; ++i) {\
			result += s[i] * b.s[i];\
		}\
		return result;\
	}\
	T lenSq() const { return dot(*this); }\
	T length() const { return (T)sqrt(lenSq()); }\
	This normalize() const { return (*this) / length(); }

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
	template<int N>\
	auto & operator()(_vec<int,N> const & i) {\
		if constexpr (N == 1) {\
			return (*this)[i(0)];\
		} else {\
			return (*this)[i(0)](i.template subset<N-1, 1>());\
		}\
	}\
	template<int N>\
	auto const & operator()(_vec<int,N> const & i) const {\
		if constexpr (N == 1) {\
			return (*this)[i(0)];\
		} else {\
			return (*this)[i(0)](i.template subset<N-1, 1>());\
		}\
	}\
\
	/* same but for std::array */\
	/* NOTICE this is only because _vec::iterator needs a std::array for indexing */\
	/*  and that is because _vec<T> can't seem to use _vec<int> for indexes, gets an 'incomplete type' error */\
	template<int N>\
	auto & operator()(std::array<int,N> const & i) {\
		if constexpr (N == 1) {\
			return (*this)[i[0]];\
		} else {\
			std::array<int,N-1> subi;\
			std::copy(i.begin() + 1, i.begin() + N, subi.begin());\
			return (*this)[i[0]](subi);\
		}\
	}\
	template<int N>\
	auto const & operator()(std::array<int,N> const & i) const {\
		if constexpr (N == 1) {\
			return (*this)[i[0]];\
		} else {\
			std::array<int,N-1> subi;\
			std::copy(i.begin() + 1, i.begin() + N, subi.begin());\
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

#define TENSOR_ADD_SUBSET_ACCESS()\
\
	/* assumes packed tensor */\
	template<int subdim, int offset>\
	_vec<T,subdim> & subset() {\
		static_assert(offset + subdim <= localDim);\
		return *(_vec<T,subdim>*)(s+offset);\
	}\
\
	/* assumes packed tensor */\
	template<int subdim, int offset>\
	_vec<T,subdim> const & subset() const {\
		static_assert(offset + subdim <= localDim);\
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

// also TODO can't use this in conjunction with the requires to_ostream or you get ambiguous operator
// the operator<< requires to_fields and _vec having fields probably doesn't help
// ... not using this
#define TENSOR_ADD_TO_OSTREAM(classname)\
	std::ostream & to_ostream(std::ostream & o) const {\
		return Common::iteratorToOStream(o, *this);\
	}

// lambda ctor
#define TENSOR_ADD_LAMBDA_CTOR(classname)\
	/* use _vec<int, rank> as our lambda index: */\
	classname(std::function<Scalar(intN)> f) {\
		auto w = write();\
		for (auto i = w.begin(); i != w.end(); ++i) {\
			*i = f(i.readIndex);\
		}\
	}\
\
	/* use (int...) as the lambda index */\
	/* since I can't just accept function(Scalar(int,...)), I need to require the type to match */\
	/* mind you in C++ I can't just say the signature is FunctionFromLambda<Lambda>::FuncType ... */\
	/* no ... I have to accept all and then requires that part */\
	template <typename Lambda>\
	classname(Lambda lambda)\
	requires (\
		std::is_same_v<\
			Common::FunctionFromLambda<Lambda>,\
			Common::FunctionFromTupleArgs<\
				Scalar,\
				Common::tuple_of_same_type<int, rank>\
			>\
		>\
	) {\
		using Func = typename Common::FunctionFromLambda<Lambda>::FuncType;\
		Func f(lambda);\
		auto w = write();\
		for (auto i = w.begin(); i != w.end(); ++i) {\
			/* *i = std::apply(f, i.readIndex); ... I need to implement some std::get etc stuff for this to work */\
			/* so until then I'll copy into a std::array<int> */\
			std::array<int, rank> iarray;\
			/*std::copy(iarray.begin(), iarray.end(), i.index.begin());  ... crashing */\
			for (int j = 0; j < rank; ++j) iarray[j] = i.readIndex[j];\
			*i = std::apply(f, iarray);\
		}\
	}

#define TENSOR_ADD_LIST_CTOR(classname)\
	classname(std::initializer_list<T> l)\
	 /* only do list constructor for non-specialized types */\
	 /*(cuz they already accept lists via matching with their ctor args) */\
	requires (localDim > 4)\
	{\
		auto src = l.begin();\
		auto dst = this->begin();\
		for (; src != l.end() && dst != this->end(); ++src, ++dst) {\
			*dst = *src;\
		}\
		for (; dst != this->end(); ++dst) {\
			*dst = {};\
		}\
	}

#define TENSOR_ADD_SCALAR_CTOR(classname)\
	classname(Scalar const & x) {\
		for (int i = 0; i < localCount; ++i) {\
			s[i] = x;\
		}\
	}

// vector cast operator
// TODO not sure how to write this to generalize into _sym and others (or if I should try to?)
// explicit 'this->s' so subclasses can use this macro (like _quat)
#define TENSOR_ADD_CTOR_FOR_GENERIC_VECTORS(classname, othername)\
	template<typename U, int N>\
	classname(othername<U, N> const & t) {\
		int i = 0;\
		for (; i < localCount && i < t.localCount; ++i) {\
			this->s[i] = (T)t.s[i];\
		}\
		for (; i < localCount; ++i) {\
			this->s[i] = {};\
		}\
	}

#define TENSOR_ADD_CTORS(classname)\
	TENSOR_ADD_SCALAR_CTOR(classname)\
	TENSOR_ADD_CTOR_FOR_GENERIC_VECTORS(classname, classname)\
	TENSOR_ADD_LAMBDA_CTOR(classname)\
	TENSOR_ADD_LIST_CTOR(classname)

/*
TODO InnerIterator (iterator i1 first) vs OuterIterator (iterator iN first)
ReadIterator vs WriteIterator
(Based on) ReadIndexIterator vs WriteIndexIterator
*/
#define TENSOR_ADD_ITERATOR(classname)\
\
	template<int i>\
	struct ReadIncInner {\
		static bool exec(intN & index) {\
			/* inc 0 first */\
			++index[i];\
			if (index[i] < This::template dim<i>) return true;\
			if (i < rank-1) index[i] = 0;\
			return false;\
		}\
	};\
\
	template<int i>\
	struct ReadIncOuter {\
		static bool exec(intN & index) {\
			/* inc n-1 first */\
			constexpr int j = rank-1-i;\
			++index[j];\
			if (index[j] < This::template dim<j>) return true;\
			if (j > 0) index[j] = 0;\
			return false;\
		}\
	};\
\
	/* read iterator */\
	/* - sized to the tensor rank (includes multiple-rank nestings) */\
	/* - range is 0's to the tensor dims() */\
	template<typename OwnerConstness>\
	struct ReadIterator {\
		using intN = typename OwnerConstness::intN;\
		OwnerConstness & owner;\
		intN index;\
		ReadIterator(OwnerConstness & owner_, intN index_ = {}) : owner(owner_), index(index_) {}\
		ReadIterator(ReadIterator const & o) : owner(o.owner), index(o.index) {}\
		ReadIterator(ReadIterator && o) : owner(o.owner), index(o.index) {}\
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
		auto & operator*() const {\
			return owner(index);\
		}\
\
		ReadIterator & operator++() {\
			Common::ForLoop<0,rank,ReadIncInner>::exec(index);\
			return *this;\
		}\
		ReadIterator & operator++(int) {\
			Common::ForLoop<0,rank,ReadIncInner>::exec(index);\
			return *this;\
		}\
\
		std::ostream & to_ostream(std::ostream & o) const {\
			return o << "ReadIterator(owner=" << &owner << ", index=" << index << ")";\
		}\
\
		static ReadIterator begin(OwnerConstness & v) {\
			return ReadIterator(v);\
		}\
		static ReadIterator end(OwnerConstness & v) {\
			intN index;\
			/* inc 0 first */\
			index[rank-1] = OwnerConstness::template dim<rank-1>;\
			/* inc n-1 first */\
			/*index[0] = OwnerConstness::template dim<0>;*/\
			/* end inc */\
			return ReadIterator(v, index);\
		}\
	};\
\
	using iterator = ReadIterator<This>;\
	iterator begin() { return iterator::begin(*this); }\
	iterator end() { return iterator::end(*this); }\
	using const_iterator = ReadIterator<This const>;\
	const_iterator begin() const { return const_iterator::begin(*this); }\
	const_iterator end() const { return const_iterator::end(*this); }\
	const_iterator cbegin() const { return const_iterator::begin(*this); }\
	const_iterator cend() const { return const_iterator::end(*this); }\
\
	template<typename WriteOwnerConstness>\
	struct Write {\
		using intW = _vec<int, numNestings>;\
		using intR = intN;\
\
		WriteOwnerConstness & owner;\
\
		Write(WriteOwnerConstness & owner_) : owner(owner_) {}\
		Write(Write const & o) : owner(o.owner) {}\
		Write(Write && o) : owner(o.owner) {}\
		Write & operator=(Write const & o) { owner = o.owner; return *this; }\
		Write & operator=(Write && o) { owner = o.owner; return *this; }\
\
		template<int i>\
		struct WriteIncInner {\
			static bool exec(intW & index) {\
				/* inc 0 first */\
				++index[i];\
				if (index[i] < This::template count<i>) return true;\
				if (i < numNestings-1) index[i] = 0;\
				return false;\
			}\
		};\
\
		template<int i>\
		struct WriteIncOuter {\
			static bool exec(intW & index) {\
				/* inc n-1 first */\
				constexpr int j = numNestings-1-i;\
				++index[j];\
				if (index[j] < This::template count<j>) return true;\
				if (j > 0) index[j] = 0;\
				return false;\
			}\
		};\
\
		/* write iterator */\
		/* - sized to the # of nestings */\
		/* - range is 0's to each nesting's .localCount */\
		template<typename OwnerConstness>\
		struct WriteIterator {\
			OwnerConstness & owner;\
			intW index;\
			intR readIndex;\
			WriteIterator(OwnerConstness & owner_, intW index_ = {})\
			: owner(owner_),\
				index(index_),\
				readIndex(Traits::getReadForWriteIndex(index)) {}\
			WriteIterator(WriteIterator const & o)\
				: owner(o.owner),\
				index(o.index),\
				readIndex(o.readIndex) {}\
			WriteIterator(WriteIterator && o)\
				: owner(o.owner),\
				index(o.index),\
				readIndex(o.readIndex) {}\
			WriteIterator & operator=(WriteIterator const & o) {\
				owner = o.owner;\
				index = o.index;\
				readIndex = o.readIndex;\
				return *this;\
			}\
			WriteIterator & operator=(WriteIterator && o) {\
				owner = o.owner;\
				index = o.index;\
				readIndex = o.readIndex;\
				return *this;\
			}\
			bool operator==(WriteIterator const & o) const {\
				return &owner == &o.owner && index == o.index;\
			}\
			bool operator!=(WriteIterator const & o) const {\
				return !operator==(o);\
			}\
			auto & operator*() {\
				/* cuz it takes less operations than by-read-index */\
				return Traits::getByWriteIndex(owner, index);\
			}\
\
			WriteIterator & operator++() {\
				Common::ForLoop<0,numNestings,WriteIncInner>::exec(index);\
				readIndex = Traits::getReadForWriteIndex(index);\
				return *this;\
			}\
			WriteIterator & operator++(int) {\
				Common::ForLoop<0,numNestings,WriteIncInner>::exec(index);\
				readIndex = Traits::getReadForWriteIndex(index);\
				return *this;\
			}\
\
			static WriteIterator begin(OwnerConstness & v) {\
				return WriteIterator(v);\
			}\
			static WriteIterator end(OwnerConstness & v) {\
				intR index;\
				/* inc 0 first */\
				index[numNestings-1] = This::template count<numNestings-1>;\
				return WriteIterator(v, index);\
			}\
		};\
\
		/* TODO if .write() was called by a const object, so WriteOwnerConstness is 'This const' */\
		/* then should .begin() .end() be allowed?  or should they be forced to be 'This const' as well? */\
		using iterator = WriteIterator<This>;\
		iterator begin() { return iterator::begin(owner); }\
		iterator end() { return iterator::end(owner); }\
		using const_iterator = WriteIterator<This const>;\
		const_iterator begin() const { return const_iterator::begin(owner); }\
		const_iterator end() const { return const_iterator::end(owner); }\
		const_iterator cbegin() const { return const_iterator::begin(owner); }\
		const_iterator cend() const { return const_iterator::end(owner); }\
	};\
\
	Write<This> write() { return Write<This>(*this); }\
	/* wait, if Write<This> write() is called by a const object ... then the return type is const ... could I detect that from within Write to forward on to Write's inner class ctor? */\
	Write<This const> write() const { return Write<This const>(*this); }\

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
	TENSOR_ADD_CTORS(classname) /* ctors, namely lambda ctor, needs read iterators*/ \
	TENSOR_ADD_ITERATOR(classname)\
	TENSOR_ADD_UNM(classname)\
	TENSOR_ADD_DOT(classname)\
	TENSOR_ADD_CMP_OP(classname)\
	TENSOR_ADD_SIZE(classname)

// only add these to _vec and specializations
// ... so ... 'classname' is always '_vec' for this set of macros
#define TENSOR_VECTOR_CLASS_OPS(classname)\
	TENSOR_ADD_VECTOR_BRACKET_INDEX()\
	TENSOR_ADD_CALL_INDEX()\
	TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_SUBSET_ACCESS()\
\
	/* assumes Inner operator* exists */\
	/* TODO name this 'product' since 'volume' is ambiguous cuz it could alos mean product-of-dims */\
	T volume() const {\
		T res = s[0];\
		for (int i = 1; i < localCount; ++i) {\
			res *= s[i];\
		}\
		return res;\
	}\
\
	/* accepts int into .s[] storage, returns _intN<localRank> of how to index it using operator() */\
	static _vec<int,localRank> getLocalReadForWriteIndex(int writeIndex) {\
		return _vec<int,localRank>{writeIndex};\
	}

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
	using Type = T;
	using Inner = void;
	using InnerTraits = void;
	static constexpr int rank = 0;
	static constexpr int numNestings = 0; // number of _vec<_vec<..._vec<T, ...>'s 
	using Scalar = T;
	template<int index> using Nested = void;
};

// recursive/vec/matrix/tensor case
template<typename T>
requires is_tensor_v<T>
struct VectorTraits<T> {
	using Type = T;
	using Inner = typename T::Inner;
	using InnerTraits = VectorTraits<Inner>;
	using intN = typename T::intN;

	// using rank = localRank + VectorTraits<Inner>::rank in _vec
	static constexpr int rank = T::rank;
	// using VectorTraits<This>::rank in _vec
	// would be nice to do all recursive calcs inside VectorTraits instead of across _vec
	// but this screws up with _tensor<real,3,3,3> rank3 and above
	//static constexpr int rank = T::localRank + InnerTraits::rank;
	
	static constexpr int numNestings = 1 + InnerTraits::numNestings;

	using Scalar = typename InnerTraits::Scalar;

	// a function for getting the i'th component of the size vector
	template<int i>
	static constexpr int calc_ith_dim() {
		static_assert(i >= 0, "you tried to get a negative size");
		static_assert(i < rank, "you tried to get an oob size");
		if constexpr (i < T::localRank) {
			return T::localDim;
		} else {
			return InnerTraits::template calc_ith_dim<i - T::localRank>();
		}
	}

	// .. then for loop iterator in dims() function and just a single exceptional case in the dims() function is needed
	static constexpr auto dims() {
		// if this is a vector-of-scalars, such that the dims would be an int1, just use int
		if constexpr (T::rank == 1) {
			return T::localDim;
		} else {
			// use an int[localDim]
			intN sizev;
#if 0
#error "TODO constexpr for loop.  I could use my template-based one, but that means one more inline'd class, which is ugly."
#else
			for (int i = 0; i < T::localRank; ++i) {
				sizev.s[i] = T::localDim;
			}
			if constexpr (T::localRank < rank) {
				// special case reading from int
				if constexpr (Inner::rank == 1) {
					sizev.s[T::localRank] = InnerTraits::dims();
				} else {
					// assigning sub-vector
					sizev.template subset<rank-T::localRank, T::localRank>()
						= InnerTraits::dims();
				}
			}
#endif
			return sizev;
		}
	}

	template<int index>
	using Nested = std::conditional_t<
		index == 0,
		T,
		typename VectorTraits<Inner>::template Nested<index - 1>
	>;

	using intW = _vec<int,numNestings>;
	
	static intN getReadForWriteIndex(intW const & i) {
		intN res;
		res.template subset<Type::localRank, 0>() = Type::getLocalReadForWriteIndex(i[0]);
		if constexpr (numNestings > 1) {
			//static_assert(rank - Type::localRank == Inner::rank);
			res.template subset<rank-Type::localRank, Type::localRank>() = InnerTraits::getReadForWriteIndex(i.template subset<numNestings-1,1>());
		}
		return res;
	}

	static Scalar & getByWriteIndex(T & t, intW const & index) {
		if constexpr (numNestings == 1) {
			return t.s[index.s[0]];
		} else {
			return InnerTraits::getByWriteIndex(t.s[index.s[0]], index.template subset<numNestings-1,1>());
		}
	}
	static Scalar const & getByWriteIndex(T const & t, intW const & index) {
		if constexpr (numNestings == 1) {
			return t.s[index.s[0]];
		} else {
			return InnerTraits::getByWriteIndex(t.s[index.s[0]], index.template subset<numNestings-1,1>());
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

	T s[localCount] = {};
	_vec() {}
	
	TENSOR_VECTOR_CLASS_OPS(_vec)
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
		T s[localCount];
	};
	_vec() {}
	_vec(T x_, T y_) : x(x_), y(y_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y)
	);

	TENSOR_VECTOR_CLASS_OPS(_vec)

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
static_assert(std::is_same_v<float2::Scalar, float>);
static_assert(std::is_same_v<float2::Inner, float>);
static_assert(float2::rank == 1);
static_assert(float2::dim<0> == 2);
static_assert(float2::numNestings == 1);
static_assert(float2::count<0> == 2);

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
		T s[localCount];
	};
	_vec() {}
	_vec(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y),
		std::make_pair("z", &This::z)
	);

	TENSOR_VECTOR_CLASS_OPS(_vec)

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
static_assert(std::is_same_v<float3::Scalar, float>);
static_assert(std::is_same_v<float3::Inner, float>);
static_assert(float3::rank == 1);
static_assert(float3::dim<0> == 3);
static_assert(float3::numNestings == 1);
static_assert(float3::count<0> == 3);

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
		T s[localCount];
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
static_assert(std::is_same_v<float4::Scalar, float>);
static_assert(std::is_same_v<float4::Inner, float>);
static_assert(float4::rank == 1);
static_assert(float4::dim<0> == 4);
static_assert(float4::numNestings == 1);
static_assert(float4::count<0> == 4);


template<int N> using boolN = _vec<bool, N>;
template<int N> using ucharN = _vec<unsigned char, N>;
template<int N> using intN = _vec<int, N>;
template<int N> using uintN = _vec<unsigned int, N>;
template<int N> using floatN = _vec<float, N>;
template<int N> using doubleN = _vec<double, N>;


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
static_assert(float2x2::rank == 2);
static_assert(float2x2::dim<0> == 2);
static_assert(float2x2::dim<1> == 2);
static_assert(float2x2::numNestings == 2);
static_assert(float2x2::count<0> == 2);
static_assert(float2x2::count<1> == 2);

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
static_assert(float2x3::rank == 2);
static_assert(float2x3::dim<0> == 2);
static_assert(float2x3::dim<1> == 3);
static_assert(float2x3::numNestings == 2);
static_assert(float2x3::count<0> == 2);
static_assert(float2x3::count<1> == 3);

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

static_assert(std::is_same_v<float4x4::Scalar, float>);
static_assert(std::is_same_v<float4x4::Inner, float4>);
static_assert(float4x4::rank == 2);
static_assert(float4x4::dim<0> == 4);
static_assert(float4x4::dim<1> == 4);
static_assert(float4x4::numNestings == 2);
static_assert(float4x4::count<0> == 4);
static_assert(float4x4::count<1> == 4);

// vector op vector, matrix op matrix, and tensor op tensor per-component operators

#define TENSOR_ADD_VECTOR_VECTOR_OP(op)\
template<typename T, int N>\
_vec<T,N> operator op(_vec<T,N> const & a, _vec<T,N> const & b) {\
	_vec<T,N> c;\
	for (int i = 0; i < N; ++i) {\
		c[i] = a[i] op b[i];\
	}\
	return c;\
}

TENSOR_ADD_VECTOR_VECTOR_OP(+)
TENSOR_ADD_VECTOR_VECTOR_OP(-)
TENSOR_ADD_VECTOR_VECTOR_OP(/)

// vector * vector
//TENSOR_ADD_VECTOR_VECTOR_OP(*) will cause ambiguous evaluation of matrix/matrix mul
// so it has to be constrained to only T == _vec<T,N>:Scalar
// c_i := a_i * b_i
template<typename T, int N>
requires std::is_same_v<typename _vec<T,N>::Scalar, T>
_vec<T,N> operator*(_vec<T, N> const & a, _vec<T,N> const & b) {
	_vec<T,N> c;
	for (int i = 0; i < N; ++i) {
		c[i] = a[i] * b[i];
	}
	return c;
}


// vector op scalar, scalar op vector, matrix op scalar, scalar op matrix, tensor op scalar, scalar op tensor operations
// need to require that T == _vec<T,N>::Scalar otherwise this will spill into vector/matrix operations
// c_i := a_i * b
// c_i := a * b_i

#define TENSOR_ADD_VECTOR_SCALAR_OP(op)\
template<typename T, int N>\
_vec<T,N> operator op(_vec<T,N> const & a, typename _vec<T,N>::Scalar const & b) {\
	_vec<T,N> c;\
	for (int i = 0; i < N; ++i) {\
		c[i] = a[i] op b;\
	}\
	return c;\
}\
template<typename T, int N>\
_vec<T,N> operator op(typename _vec<T,N>::Scalar const & a, _vec<T,N> const & b) {\
	_vec<T,N> c;\
	for (int i = 0; i < N; ++i) {\
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
requires std::is_same_v<typename _mat<T,dim1,dim2>::Scalar, T>
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
requires std::is_same_v<typename _mat<T,dim1,dim2>::Scalar, T>
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
requires std::is_same_v<typename _mat<T,dim1,dim2>::Scalar, T>
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

// TODO GLSL sucks, rename this to something that makes sense, and shorthand those other longwinded GLSL names like "inverse"=>"inv", "determinant"=>"det", "transpose"=>"tr" "normalize"=>"unit"
template<typename T>
T matrixCompMult(T const & a, T const & b) {
	return T([&](T::intN i) -> typename T::Scalar {
		return a(i) * b(i);
	});
}

// vector functions
// TODO for these, should I call into the member function?

// c := a_i * b_i
// TODO generalize
template<typename T, int N>
requires std::is_same_v<typename _vec<T,N>::Scalar, T>
T dot(_vec<T,N> const & a, _vec<T,N> const & b) {
	T sum = {};
	for (int i = 0; i < N; ++i) {
		sum += a[i] * b[i];
	}
	return sum;
}

template<typename T, int N>
T lenSq(_vec<T,N> const & v) {
	return dot(v,v);
}

template<typename T, int N>
T length(_vec<T,N> const & v) {
	return sqrt(lenSq(v));
}

template<typename T, int N>
T distance(_vec<T,N> const & a, _vec<T,N> const & b) {
	return length(b - a);
}

template<typename T, int N>
_vec<T,N> normalize(_vec<T,N> const & v) {
	return v / length(v);
}

// c_i := ε_ijk * b_j * c_k
template<typename T>
requires std::is_same_v<typename _vec3<T>::Scalar, T>
_vec3<T> cross(_vec3<T> const & a, _vec3<T> const & b) {
	return _vec3<T>(
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]);
}

// c_ij := a_i * b_j
// TODO generalize to tensors c_i1_..ip_j1_..jq = a_i1..ip * b_j1..jq
template<typename T, int dim1, int dim2>
requires std::is_same_v<typename _vec<T,dim1>::Scalar, T>
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

template<typename T, int N>
_mat<T,N,N> diagonalMatrix(_vec<T,N> const & v) {
	_mat<T,N,N> a;
	for (int i = 0; i < N; ++i) {
		a[i][i] = v[i];
	}
	return a;
}

// symmetric matrices

#define TENSOR_SYMMETRIC_MATRIX_HEADER(localDim_)\
	static constexpr int localDim = localDim_;\
	static constexpr int localRank = 2;\
	static constexpr int localCount = (localDim * (localDim + 1)) >> 1;

template<int N>
int symIndex(int i, int j) {
	if (j > i) return symIndex<N>(j,i);
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

#define TENSOR_ADD_SYMMETRIC_MATRIX_CALL_INDEX(classname)\
\
	/* a(i,j) := a_ij = a_ji */\
	/* this is the direct acces */\
	Inner & operator()(int i, int j) { return s[symIndex<localDim>(i,j)]; }\
	Inner const & operator()(int i, int j) const { return s[symIndex<localDim>(i,j)]; }\
\
	struct Accessor {\
		classname & owner;\
		int i;\
		Accessor(classname & owner_, int i_) : owner(owner_), i(i_) {}\
		Inner & operator[](int j) { return owner(i,j); }\
		TENSOR_ADD_CALL_INDEX()\
	};\
	Accessor operator[](int i) { return Accessor(*this, i); }\
\
	struct ConstAccessor {\
		classname const & owner;\
		int i;\
		ConstAccessor(classname const & owner_, int i_) : owner(owner_), i(i_) {}\
		Inner const & operator[](int j) { return owner(i,j); }\
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
	TENSOR_SYMMETRIC_ADD_RECURSIVE_CALL_INDEX()\

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
	TENSOR_ADD_SYMMETRIC_MATRIX_CALL_INDEX(classname)\
\
	static _vec<int,localRank> getLocalReadForWriteIndex(int writeIndex) {\
		_vec<int,localRank> readIndex;\
		int w = writeIndex+1;\
		for (int i = 1; w > 0; ++i) {\
			++readIndex(0);\
			w -= i;\
		}\
		--readIndex(0);\
		readIndex(1) = writeIndex - readIndex(0) * (readIndex(0) + 1) / 2;\
		return readIndex;\
	}

template<typename T, int dim_>
struct _sym {
	using This = _sym;
	TENSOR_SYMMETRIC_MATRIX_HEADER(dim_)
	TENSOR_HEADER()

	T s[localCount] = {};
	_sym() {}

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(_sym)
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
		T s[localCount];
	};
	_sym() {}
	_sym(T xx_, T xy_, T yy_) : xx(xx_), xy(xy_), yy(yy_) {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("xx", &This::xx),
		std::make_pair("xy", &This::xy),
		std::make_pair("yy", &This::yy)
	);

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(_sym)
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
		T s[localCount];
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
		T s[localCount];
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
};

// symmetric op symmetric

#define TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(op)\
template<typename T, int N>\
_sym<T,N> operator op(_sym<T,N> const & a, _sym<T,N> const & b) {\
	_sym<T,N> c;\
	for (int i = 0; i < c.localCount; ++i) {\
		c.s[i] = a.s[i] op b.s[i];\
	}\
	return c;\
}

TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(+)
TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(-)
TENSOR_ADD_SYMMETRIC_SYMMETRIC_OP(/)

// symmetric op scalar, scalar op symmetric

#define TENSOR_ADD_SYMMETRIC_MATRIX_SCALAR_OP(op)\
template<typename T, int N>\
requires std::is_same_v<typename _sym<T,N>::Scalar, T>\
_sym<T,N> operator op(_sym<T,N> const & a, T const & b) {\
	_sym<T,N> c;\
	for (int i = 0; i < c.localCount; ++i) {\
		c.s[i] = a.s[i] op b;\
	}\
	return c;\
}\
template<typename T, int N>\
requires std::is_same_v<typename _sym<T,N>::Scalar, T>\
_sym<T,N> operator op(T const & a, _sym<T,N> const & b) {\
	_sym<T,N> c;\
	for (int i = 0; i < c.localCount; ++i) {\
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
static_assert(std::is_same_v<typename float2s2::Scalar, float>);
static_assert(float2s2::rank == 2);
static_assert(float2s2::dim<0> == 2);
static_assert(float2s2::dim<1> == 2);
static_assert(float2s2::numNestings == 1);
static_assert(float2s2::count<0> == 3);

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
static_assert(std::is_same_v<typename float3s3::Scalar, float>);
static_assert(float3s3::rank == 2);
static_assert(float3s3::dim<0> == 3);
static_assert(float3s3::dim<1> == 3);
static_assert(float3s3::numNestings == 1);
static_assert(float3s3::count<0> == 6);

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
static_assert(std::is_same_v<typename float4s4::Scalar, float>);
static_assert(float4s4::rank == 2);
static_assert(float4s4::dim<0> == 4);
static_assert(float4s4::dim<1> == 4);
static_assert(float4s4::numNestings == 1);
static_assert(float4s4::count<0> == 10);

// ostream
// _vec does have .fields
// and I do have my default .fields ostream
// but here's a manual override anyways
// so that the .fields vec2 vec3 vec4 and the non-.fields other vecs all look the same
template<typename T, int N>
std::ostream & operator<<(std::ostream & o, Tensor::_vec<T,N> const & t) {
	char const * sep = "";
	o << "{";
	for (int i = 0; i < t.localCount; ++i) {
		o << sep << t.s[i];
		sep = ", ";
	}
	o << "}";
	return o;
}

// TODO print as fields of .sym, or print as vector?
template<typename T, int N>
std::ostream & operator<<(std::ostream & o, Tensor::_sym<T,N> const & t) {
	char const * sep = "";
	o << "{";
	for (int i = 0; i < t.localCount; ++i) {
		o << sep << t.s[i];
		sep = ", ";
	}
	o << "}";
	return o;
}

} // namespace Tensor

namespace std {

// tostring 

template<typename T, int n>
std::string to_string(Tensor::_vec<T, n> const & x) {
	return Common::objectStringFromOStream(x);
}

#if 0	// half baked idea of making std::apply compatible with Tensor::_vec
// tuple_size, make this match array i.e. storage size, returns localCount

template<typename T, int N>
struct tuple_size<Tensor::_vec<T,N>> { 
	static constexpr auto value = Tensor::_vec<T,N>::localCount;
};

template<typename T, int N>
struct tuple_size<Tensor::_sym<T,N>> { 
	static constexpr auto value = Tensor::_sym<T,N>::localCount; 	// (N*(N+1))/2
};

// std::get .... all the dif impls ... 

template<std::size_t I, typename T, std::size_t N> constexpr T & get(Tensor::_vec<T,N> & v) noexcept { return v[I]; }
template<std::size_t I, typename T, std::size_t N> constexpr T && get(Tensor::_vec<T,N> && v) noexcept { return v[I]; }
template<std::size_t I, typename T, std::size_t N> constexpr T const & get(Tensor::_vec<T,N> const & v) noexcept { return v[I]; }
template<std::size_t I, typename T, std::size_t N> constexpr T const && get(Tensor::_vec<T,N> const && v) noexcept { return v[I]; }

#endif

}
