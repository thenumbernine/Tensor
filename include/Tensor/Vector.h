#pragma once

/*
NEW VERSION
- no more template metaprograms, instead constexprs
- no more helper structs, instead requires
- no more lower and upper
- modeled around glsl
- maybe extensions into matlab syntax (nested tensor implicit * contraction mul)
- math indexing.  A.i.j := a_ij, which means row-major storage (sorry OpenGL)


alright conventions, esp with my sym being in the mix ...
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


if I do row-major then
	- C bracket ctor is in the same layout as the matrix
	- C notation matches matrix notation : A[i0][i1] = A_i0_i1
	- memory layout is transposed from nested index order: A_i0_i1 = A[i1 + i0 * size0]
if I do column-major then C inline indexing is transposed
	- C bracket ctor is transposed from the layout of the matrix
	- matrix notation is transposed: A[j][i] == A_ij
	- memory layout matches nested index order: A_i0_i1 = A[i0 + size0 * i1]
	- OpenGL uses this too.
...Row Major it is.

*/

#include "Tensor/Vector.h.h"	//forward-declarations better match
#include "Tensor/Math.h.h"		//tensor math functions that I've got implemented as members  as well
#include "Tensor/Inverse.h.h"
#include "Tensor/Index.h.h"
#include "Tensor/Range.h.h"
#include "Tensor/AntiSymRef.h"
#include "Tensor/Meta.h"
#include "Common/String.h"
#include "Common/Function.h"	//FunctionFromLambda
#include "Common/Sequence.h"	//seq_reverse_t, make_integer_range
#include "Common/Exception.h"
#include "Common/Test.h"		//STATIC_TEST_EQ
#include <tuple>
#include <array>
#include <functional>	//reference_wrapper, also function<> is by Partial
#include <cmath>		//sqrt()

#ifdef DEBUG
#define TENSOR_USE_BOUNDS_CHECKING
#endif

#ifdef TENSOR_USE_BOUNDS_CHECKING
#define TENSOR_INSERT_BOUNDS_CHECK(index)\
	if (index < 0 || index >= localCount) throw Common::Exception() << "tried to read oob index " << index << " for size " << localCount << " type " << typeid(This).name();
#else
#define TENSOR_INSERT_BOUNDS_CHECK(index)
#endif

namespace Tensor {

constexpr int consteval_symmetricSize(int d, int r) {
	return nChooseR(d + r - 1, r);
}

constexpr int consteval_antisymmetricSize(int d, int r) {
	return nChooseR(d, r);
}


template<typename T>
using RepPtrByLocalRank = Common::tuple_rep_t<T, std::remove_pointer_t<T>::localRank>;

//TupleToSeqMap can't handle templated-integrals
// so you have to pass it a class with a 'integral value' static constexpr instead.
template<typename T>
struct GetPtrLocalDim {
	static constexpr int value = std::remove_pointer_t<T>::localDim;
};

template<typename T>
struct GetPtrLocalCount {
	static constexpr int value = std::remove_pointer_t<T>::localCount;
};

template<typename T>
using GetPtrLocalStorage = typename std::remove_pointer_t<T>::LocalStorage;

template<
	int dim,
	int rank,
	template<int> typename storageRank2,
	template<int, int> typename storageRankN
>
struct GetTupleWrappingStorageForRankImpl {
	static_assert(rank >= 0);
	static constexpr auto value() {
		if constexpr (rank == 0) {
			return std::tuple<>();
		} else if constexpr (rank == 1) {
			return std::tuple<storage_vec<dim>>();
		} else if constexpr (rank == 2) {
			return std::tuple<storageRank2<dim>>();
		} else {
			return std::tuple<storageRankN<dim, rank>>();
		}
	}
	using type = decltype(value());
};
template<
	int dim,
	int rank,
	template<int> typename storageRank2,
	template<int, int> typename storageRankN
>
using GetTupleWrappingStorageForRank = typename GetTupleWrappingStorageForRankImpl<dim, rank, storageRank2, storageRankN>::type;

// Template<> is used for rearranging internal structure when performing linear operations on tensors
// Template can't go in TENSOR_HEADER cuz Quat<> uses TENSOR_HEADER and doesn't fit the template form
#define TENSOR_THIS(classname)\
	using This = classname;\
	static constexpr bool isTensorFlag = true;\
	static constexpr bool dontUseFieldsOStream = true;

#define TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, localRank_)\
\
	/*  This is the next most nested class, so vector-of-vector is a matrix. */\
	using Inner = Inner_;\
\
	/* this is this particular dimension of our vector */\
	/* M = matrix<T,i,j> == vec<vec<T,j>,i> so M::localDim == i and M::Inner::localDim == j  */\
	/*  i.e. M::dims() = int2(i,j) and M::dim<0> == i and M::dim<1> == j */\
	static constexpr int localDim = localDim_;\
\
	/*how much does this structure contribute to the overall rank. */\
	/* for vec it is 1, for sym it is 2 */\
	static constexpr int localRank = localRank_;

// used by quat
#define TENSOR_TEMPLATE_T(classname)\
\
	template<typename Inner2>\
	using Template = classname<Inner2>;\
\
	template<typename NewInner>\
	using ReplaceInner = Template<NewInner>;\
\
	template<int newLocalDim>\
	using ReplaceLocalDim = Template<Inner>;

#define TENSOR_TEMPLATE_T_I(classname)\
\
	template<typename Inner2, int localDim2>\
	using Template = classname<Inner2, localDim2>;\
\
	template<typename NewInner>\
	using ReplaceInner = Template<NewInner, localDim>;\
\
	template<int newLocalDim>\
	using ReplaceLocalDim = Template<Inner, newLocalDim>;

// used for ExpandIthIndex.  all tensors use tensorr, except symR and asymR can use symR and asymR.
#define TENSOR_EXPAND_TEMPLATE_TENSORR()\
\
	template<int index>\
	requires (index >= 0 && index < localRank)\
	using ExpandLocalStorage = Common::tuple_rep_t<storage_vec<localDim>, localRank>;

/*
This contains definitions of types and values used in all tensors.
Defines the following:
Scalar
rank
intN
numNestings
intW
ReplaceNested<int, typename>
ExpandIthIndex<int> ... requires numNestingsToIndex<int>
ExpandIndex<int...>
ExpandIndexSeq<std::integer_sequence<T, T...>>
ExpandAllIndexes<>
RemoveIthNesting<int>
RemoveIthIndex<int>
RemoveIndex<int...>
RemoveIndexSeq<std::integer_sequence<T, T...>>
ReplaceDim<int, int>
Nested<int>
count<int>
numNestingsToIndex<int>
InnerForIndex<int>
dim<int>
dims
ReplaceScalar<typename>

new idea
NestedPtrTuple = tuple of all nested classes, as ptrs so I can use This while defining it within the class.
Nested<i> = i'th getter of NestedPtrTuple
numNestings = tuple_size_v<NestedPtrTuple> - 1 for scalar
dimseq = each NestedPtrTuple's localDim => seq_rep x localRank => apply to seq_cat_t
rank = tuple_size_v of dimseq
dim<i> = 'th getter of dimseq
Scalar = NestedPtrTuple's last
*/
#define TENSOR_HEADER()\
\
	/* TENSOR_HEADER goes right after the struct-specific header which defines localCount...*/\
	static_assert(localCount > 0);\
\
	/* The types have to be ptrs so I can use 'This' class */\
	/* Would be nice to just collect all inner-based templates in a tuple, and have something for succively applying them to produce this tuple. */\
	/* It would have to excludes Scalar cuz thats the arg -- pass Scalar separately into the tuple-application mechanism */\
	/* Sadly a template can't go in a tuple ... so intead you gotta just map the NestedPtrTuple's types's ReplaceInner<> templates when you want them */\
	struct NestedPtrTupleImpl {\
		static constexpr auto value() {\
			if constexpr (is_tensor_v<Inner>) {\
				return std::tuple_cat(\
					std::tuple<This*>(),\
					Inner::NestedPtrTupleImpl::value()\
				);\
			} else {\
				/* while I am at the base of the chain, grab Scalar*/\
				return std::tuple<This*, Inner*>();\
			}\
		}\
		using type = decltype(value());\
	};\
	using NestedPtrTuple = typename NestedPtrTupleImpl::type;\
	/* can't do this, because first type is This, and we're still in This */\
	/*using NestedTuple = Common::TupleTypeMap<NestedPtrTupleImpl, std::remove_pointer_t>;*/\
\
	/* for when you just want to work with the nested tensors, and not the final scalar */\
	using NestedPtrTensorTuple = Common::tuple_remove_last_t<NestedPtrTuple>;\
\
	using Scalar = typename std::remove_pointer_t<std::tuple_element_t<std::tuple_size_v<NestedPtrTuple>-1, NestedPtrTuple>>;\
\
	/* how many vec<vec<...'s there are ... no extra +'s for multi-rank nestings */\
	/* so sym<> has numNestings=1 and rank=2, vec<> has numNestings=1 and rank=1, vec<vec<>>==mat<> has numNestings=2 and rank=2 */\
	static constexpr int numNestings = std::tuple_size_v<NestedPtrTensorTuple>;\
\
	/* Get the i'th nested type */\
	template<int i>\
	using Nested = typename std::remove_pointer_t<std::tuple_element_t<i, NestedPtrTuple>>;\
\
	using countseq = Common::TupleToSeqMap<int, NestedPtrTensorTuple, GetPtrLocalCount>;\
\
	/* Get the i'th nesting's count aka storage size */\
	template<int i>\
	static constexpr int count = Common::seq_get_v<i, countseq>;\
	/*static constexpr int count = Nested<i>::localCount;*/\
\
	static constexpr int totalCount = Common::seq_multiplies(countseq());\
\
	/* same idea as in NestedPtrTensorTuple, but members are duplicated for their localRank */\
	/* so that it is correlated with tensor index instead of nesting */\
	using InnerPtrTensorTuple = Common::tuple_apply_t<Common::tuple_cat_t, Common::TupleTypeMap<NestedPtrTensorTuple, RepPtrByLocalRank>>;\
\
	/* Just like NestedPtrTuple vs NestedPtrTensorTuple, */\
	/* and for the sake of InnerForIndex, I'm appending Scalar once again */\
	using InnerPtrTuple = Common::tuple_cat_t<InnerPtrTensorTuple, std::tuple<Scalar*>>;\
\
	/* TODO rename 'Inner' => 'LocalInner' and rename 'InnerForIndex' to just 'Inner' ? */\
	template<int index>\
	using InnerForIndex = typename std::remove_pointer_t<std::tuple_element_t<(size_t)index, InnerPtrTuple>>;\
\
	/* int sequence of dimensions */\
	using dimseq = Common::TupleToSeqMap<int, InnerPtrTensorTuple, GetPtrLocalDim>;\
\
	template<int index>\
	static constexpr int dim = Common::seq_get_v<index, dimseq>;\
	/*static constexpr int dim = InnerForIndex<index>::localDim;*/\
\
	/*  this is the rank/degree/index/number of letter-indexes of your tensor. */\
	/*  for vectors-of-vectors this is the nesting. */\
	/*  if you use any (anti?)symmetric then those take up 2 ranks / 2 indexes each instead of 1-rank each. */\
	static constexpr int rank = std::tuple_size_v<InnerPtrTensorTuple>;\
\
	/* used for vector-dereferencing into the tensor. */\
	using intN = vec<int,rank>;\
\
	/* used for write-iterators and converting write-iterator indexes to read index vectors */\
	using intW = vec<int,numNestings>;\
\
	/* used for getLocalReadForWriteIndex() return type */\
	/* and ofc the write local index is always 1 */\
	using intNLocal = vec<int,localRank>;\
\
	/* This needs to be a function because if it was a constexpr value then the compiler would complain that DimsImpl is using vec<int,1> before it is defined.*/\
	/* I can circumvent that error by having rank-1 vec's have a dims == int instead of vec<int> ... and then i get dims as a value instead of function, and all is well */\
	/*  except that makes me need to write conditional code everywhere for rank-1 and rank>1 dims */\
	static constexpr intN dims() { return intN(dimseq()); }\
\
	/* TODO alternative for indexForNesting: */\
	/* get a sequence of the local-rank for the respective nesting ... then make a sequence of the sum-of-sequence */\
	/*using NestedLocalRankSeq = Common::TupleTypeMaps<NestedPtrTensorTuple, RemovePtr, GetLocalRank>*/\
	/*using NestedLocalRankSeq = Common::tuple_apply_t<seq_cat_t, tuple cat <int> with SeqMapToType<std::make_index_sequence<numNestings>, RepSeqByCount>>*/\
	/*using IndexForNestingSeq = seq_cat_t< 0's x localRank0, 1's x localRank1, ... >;*/\
	/* Get the first index that this nesting represents. */\
	/* Same as the sum of all prior nestings' localRanks */\
	template<int nest>\
	struct IndexForNestingImpl {\
		static constexpr int value() {\
			if constexpr (nest == 0) {\
				return 0;\
			} else if constexpr (nest == 1) {\
				return localRank;\
			} else {\
				return localRank + Inner::template IndexForNestingImpl<nest-1>::value();\
			}\
		}\
	};\
	template<int nest>\
	static constexpr int indexForNesting = IndexForNestingImpl<nest>::value();\
\
	/* This == Common::tuple_apply_t<tensori, Common::tuple_cat_t<std::tuple<Scalar>, StorageTuple>>) */\
	using StorageTuple = Common::TupleTypeMap<NestedPtrTensorTuple, GetPtrLocalStorage>;\
	STATIC_ASSERT_EQ((std::tuple_size_v<StorageTuple>), numNestings);\
\
	/* notice this won't be true for Accessors */\
	/*static_assert(std::is_same_v<tensorScalarTuple<Scalar, StorageTuple>, This>);*/\
\
	/* the rest of these, I can make easier if I store a tuple of the tensori index storage helpers per class */\
	/* then provide a method for rebuiding the tensor from its tupel of strage helpers*/\
	/* and then the rest of these can be written in terms of tuple manipulation */\
\
	/* This is "Replace Inner of Nesting 'i' with NewType */\
	template<int i, typename NewType>\
	using ReplaceNested = Common::tuple_apply_t<tensori, Common::tuple_cat_t<std::tuple<NewType>, Common::tuple_subset_t<StorageTuple, 0, i>>>;\
\
	/* get the number of nestings to the j'th index */\
	template<int index>\
	struct NestingForIndexImpl {\
		static constexpr int value() {\
			static_assert(index >= 0 && index <= rank);\
			/* to get the scalar-most nesting, query rank */\
			if constexpr (index == rank) {\
				return numNestings;\
			} else if constexpr (index < localRank) {\
				return 0;\
			} else {\
				using Impl = typename Inner::template NestingForIndexImpl<index - localRank>;\
				return 1 + Impl::value();\
			}\
		}\
	};\
	template<int index>\
	static constexpr int numNestingsToIndex = NestingForIndexImpl<index>::value();\
\
	template<int index>\
	requires (index >= 0 && index < rank)\
	struct ExpandIthStorageImpl {\
		static constexpr int nest = numNestingsToIndex<index>;\
		using type = Common::tuple_insert_t<\
			Common::tuple_remove_t<nest, StorageTuple>,\
			nest,\
			typename Nested<nest>::template ExpandLocalStorage<\
				index - indexForNesting<nest>\
			>\
		>;\
	};\
	template<int index>\
	requires (index >= 0 && index < rank)\
	using ExpandIthIndexStorage = typename ExpandIthStorageImpl<index>::type;\
\
	/* expand the storage of the i'th index */\
	/* produce a tuple of new storages and then just wedge that into the tuple and rebuilt */\
	/* impl is only here cuz i would need to move ExpandLocalStorage up above this otherwise */\
	/* and that would mean splitting the HEADER and putting it in between this and other stuf above */\
	template<int index>\
	requires (index >= 0 && index < rank)\
	using ExpandIthIndex = tensorScalarTuple<Scalar, ExpandIthIndexStorage<index>>;\
\
	template<typename ThisDefer, typename Seq>\
	struct ExpandIndexSeqImpl;\
	template<typename ThisDefer, int i1, int... is>\
	struct ExpandIndexSeqImpl<ThisDefer, std::integer_sequence<int, i1, is...>> {\
		using expanded = typename ThisDefer::template ExpandIthIndex<i1>;\
		using type = typename expanded::template ExpandIndexSeqImpl<expanded, std::integer_sequence<int, is...>>::type;\
	};\
	/* gcc has this error for template<> specializations within a class: "explicit specialization in non-namespace scope" */\
	/* to fix it? just pad useless template parameters to prevent fully-specialized templates in the class */\
	template<typename ThisDefer>\
	struct ExpandIndexSeqImpl<ThisDefer, std::integer_sequence<int>> {\
		using type = ThisDefer;\
	};\
	template<typename Seq, typename ThisDefer = This>\
	using ExpandIndexSeq = typename ExpandIndexSeqImpl<ThisDefer, Seq>::type;\
\
	template<int... is>\
	using ExpandIndex = ExpandIndexSeq<std::integer_sequence<int, is...>>;\
\
	template<int deferRank = rank> /* evaluation needs to be deferred */\
	using ExpandAllIndexes = ExpandIndexSeq<std::make_integer_sequence<int, deferRank>>;\
\
	decltype(auto) expand() const {\
		return ExpandAllIndexes<>(*this);\
	}\
\
	/* yet another one where tensor type manip is easier than storage manip */\
	/* in this case, duplicating the storage by their rank means knowing the local-rank, which isn't in the storage classes */\
	template<int deferRank = rank>\
	using ExpandAllStorage = typename ExpandAllIndexes<deferRank>::StorageTuple;\
\
	template<int i>\
	requires (i >= 0 && i < numNestings)\
	using RemoveIthNestedStorage = Common::tuple_remove_t<i, StorageTuple>;\
\
	template<int i>\
	requires (i >= 0 && i < numNestings)\
	using RemoveIthNesting = tensorScalarTuple<Scalar, RemoveIthNestedStorage<i>>;\
\
	/* ok despite RemoveIthNestingStorage being up there, RemoveIthIndex will still need to know the rank and index placement info of the tensor with storage expanded and removed ... */\
	/*  so I don't think there's an easy way to do this just by storage manipulation ... */\
	template<int i>\
	requires (i >= 0 && i < rank)\
	struct RemoveIthIndexImpl {\
		using ithIndexExpanded = This::template ExpandIthIndex<i>;\
		using type = typename ithIndexExpanded::template RemoveIthNesting<\
			ithIndexExpanded::template numNestingsToIndex<i>\
		>;\
	};\
	/* same as Expand but now with RemoveIthIndex, RemoveIndex, RemoveIndexSeq */\
	template<int i>\
	requires (i >= 0 && i < rank)\
	using RemoveIthIndex =  typename RemoveIthIndexImpl<i>::type;\
\
	/* so whereas other functions are manip-storage-first, rebuild-tensor-next, */\
	/*  this one is more change-tensor-first, get-storage-next */\
	template<int i>\
	requires (i >= 0 && i < rank)\
	using RemoveIthIndexStorage = typename RemoveIthIndex<i>::StorageTuple;\
\
	/* RemoveIndexSeqImpl assumes Seq is a integer_sequence<int, ...> */\
	/*  and assumes it is already sorted in descending order. */\
	template<typename ThisDefer, typename Seq>\
	struct RemoveIndexSeqImpl;\
	template<typename ThisDefer, int i1, int... is>\
	struct RemoveIndexSeqImpl<ThisDefer, std::integer_sequence<int, i1, is...>> {\
		using removed = typename ThisDefer::template RemoveIthIndex<i1>;\
		using type = typename removed::template RemoveIndexSeqImpl<removed, std::integer_sequence<int, is...>>::type;\
	};\
	template<typename ThisDefer, int i1>\
	struct RemoveIndexSeqImpl<ThisDefer, std::integer_sequence<int, i1>> {\
		using type = RemoveIthIndex<i1>;\
	};\
	template<typename ThisDefer>\
	struct RemoveIndexSeqImpl<ThisDefer, std::integer_sequence<int>> {\
		using type = ThisDefer;\
	};\
	/* RemoveIndexSeq sorts the list, so you don't need to worry about order of arguments */\
	template<typename Seq, typename ThisDefer = This>\
	using RemoveIndexSeq = typename RemoveIndexSeqImpl<\
		ThisDefer,\
		Common::seq_reverse_t<\
			Common::seq_sort_t<\
				Seq\
			>\
		>\
	>::type;\
\
	template<int... is>\
	using RemoveIndex = RemoveIndexSeq<std::integer_sequence<int, is...>>;\
\
	/* This along with RemoveIthIndexStorage suffer from the case that, even if you find and remove a storage component, */\
	/*  you still need to know nesting<->index information on the subsequent tensor, so storage-manipulation isn't as feesible as simply making the new tensor with exapnded removed storage, then manipulating that storage. */\
	template<int index, int newDim>\
	requires (index >= 0 && index < rank)\
	struct ReplaceDimImpl {\
		static constexpr auto value() {\
			if constexpr (This::template dim<index> == newDim) {\
				return (This*)nullptr;\
			} else {\
				using expanded = typename This\
					::template ExpandIndex<index>;\
				using type = typename expanded\
					::template ReplaceNested<\
						expanded::template numNestingsToIndex<index>,\
						typename expanded\
							::template InnerForIndex<index>\
							::template ReplaceLocalDim<newDim>\
					>;\
				return (type*)nullptr;\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<int index, int newDim>\
	requires (index >= 0 && index < rank)\
	using ReplaceDim = typename ReplaceDimImpl<index, newDim>::type;\
\
	/* isSquare means all dims match */\
	struct IsSquareImpl {\
		static constexpr bool value() {\
			if constexpr (is_tensor_v<Inner>) {\
				return Inner::isSquare && Inner::localDim == localDim;\
			} else {\
				return true;\
			}\
		}\
	};\
	static constexpr bool isSquare = IsSquareImpl::value();\
\
	template<typename NewScalar>\
	using ReplaceScalar = tensorScalarTuple<NewScalar, StorageTuple>;\
\
	/* not sure about this one ... */\
	/* I'm suign it with TensorSumResult to expand only the rank that matches this storage */\
	/*  but to do so expanding in sequential order for the sake of the ExpandIthIndex optimizations of symR and asymR */\
	/* TODO ... 'unpack-and-apply'  template that goes down the Nestings?*/ \
	template<typename O>\
	struct ExpandMatchingLocalRankImpl {\
		static constexpr auto value() {\
			/* do I need this condition? */\
			if constexpr (rank == localRank) {\
				return (Inner*)nullptr;\
			} else {\
				/* expand the matching 0 and 1 indexes in the 'other' type */\
				/* TODO right now I have symR Expand optimized to symR if expanding the end indexes */\
				/*  so this means make sure to expand the 1st index 1st or you might risk turning symR into tensorr */\
				using O2 = typename O::template ExpandIndexSeq<std::make_integer_sequence<int, localRank>>;\
				/* then get its inner past the expanded index (and apply it to the recursive case) */\
				return (typename Inner::template TensorSumResult<typename O2::template InnerForIndex<localRank>>*)nullptr;\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<typename O>\
	using ExpandMatchingLocalRank = typename ExpandMatchingLocalRankImpl<O>::type;

#if 0 // hmm, as a member ReplaceWithZero is choking
	/* once again can't operate on storage cuz it doesn't store rank info */\
	template<int deferRank = rank>\
	struct ReplaceStorageWithZeroImpl {\
	};
	template<int deferRank = rank>\
	using ReplaceStorageWithZero = typename ExpandAllStorage<deferRank>::template ReplaceWithZeroImpl<deferRank>::type;

	template<int deferRank = rank>\
	struct ReplaceWithZeroImpl {\
		static constexpr auto value() {\
			if constexpr (deferRank== 1) {\
				return (zero<Inner, localDim>*)nullptr;\
			} else {\
				return (zero<typename Inner::template ReplaceWithZeroImpl<>, localDim>>*)nullptr;\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<int deferRank = rank>\
	using ReplaceWithZero = typename ExpandAllIndexes<deferRank>::template ReplaceWithZeroImpl<deferRank>::type;
#endif

//for giving operators to tensor classes
//how can you add correctly-typed ops via crtp to a union?
//unions can't inherit.
//until then...

#define TENSOR_ADD_VECTOR_OP_EQ(op)\
	constexpr This & operator op(This const & b) {\
			/* sequences */\
		/*return [&]<size_t ... k>(std::index_sequence<k...>) constexpr -> This & {\
			return ((s[k] op b.s[k]), ..., *this);\
		}(std::make_index_sequence<localCount>{});*/\
			/* for-loops */\
		for (int k = 0; k < localCount; ++k) {\
			s[k] op b.s[k];\
		}\
		return *this;\
	}

#define TENSOR_ADD_SCALAR_OP_EQ(op)\
	constexpr This & operator op(Scalar const & b) {\
			/* sequences */\
		/*return [&]<size_t ... k>(std::index_sequence<k...>) constexpr -> This & {\
			return ((s[k] op b), ..., *this);\
		}(std::make_index_sequence<localCount>{});*/\
			/* for-loops */\
		for (int k = 0; k < localCount; ++k) {\
			s[k] op b;\
		}\
		return *this;\
	}

// for comparing like types, use the member operator==, because it is constexpr
// for non-like tensors there's a non-member non-constexpr below
#define TENSOR_ADD_CMP_OP()\
	constexpr bool operator==(This const & b) const {\
		return [&]<size_t ... k>(std::index_sequence<k...>) constexpr -> bool {\
			return ((s[k] == b.s[k]) && ... && (true));\
		}(std::make_index_sequence<localCount>{});\
	}\
	constexpr bool operator!=(This const & b) const {\
		return !operator==(b);\
	}

#define TENSOR_ADD_UNARY(op)\
	constexpr This operator op() const {\
		This result;\
			/* sequences */\
		/*return [&]<size_t ... k>(std::index_sequence<k...>) constexpr -> This {\
			return ((result.s[k] = op s[k]), ..., result);\
		}(std::make_index_sequence<localCount>{});*/\
			/* for-loops */\
		for (int k = 0; k < localCount; ++k) {\
			result.s[k] = op s[k];\
		}\
		return result;\
	}

// for rank-1 objects (vec, sym::Accessor, asym::Accessor)
// I'm using use operator(int) as the de-facto for rank-1, operator(int,int) for rank-1 etc
#define TENSOR_ADD_VECTOR_CALL_INDEX_PRIMARY()\
\
	/* a(i) := a_i */\
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int const i) {\
		TENSOR_INSERT_BOUNDS_CHECK(i);\
		return s[i];\
	}\
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int const i) const {\
		TENSOR_INSERT_BOUNDS_CHECK(i);\
		return s[i];\
	}

#define TENSOR_ADD_BRACKET_FWD_TO_CALL()\
	/* a[i] := a_i */\
	/* operator[int] calls through operator(int) */\
	constexpr decltype(auto) operator[](int const i) { return (*this)(i); }\
	constexpr decltype(auto) operator[](int const i) const { return (*this)(i); }

//operator[int] forwards to operator(int)
//operator(int...) forwards to operator(int)(int...)
#define TENSOR_ADD_RANK1_CALL_INDEX_AUX()\
\
	TENSOR_ADD_BRACKET_FWD_TO_CALL()\
\
	/* a(i1,i2,...) := a_i1_i2_... */\
	/* operator(int, int...) calls through operator(int) */\
	template<typename Int, typename... Ints>\
	requires (rank > 1 && ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int>)))\
	constexpr decltype(auto) operator()(Int const i, Ints const ... is) {\
		return (*this)(i)(is...);\
	}\
\
	template<typename Int, typename... Ints>\
	requires (rank > 1 && ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int>)))\
	constexpr decltype(auto) operator()(Int const i, Ints const ... is) const {\
		return (*this)(i)(is...);\
	}

// operator(vec<int,...>) forwards to operator(int...)
#define TENSOR_ADD_INT_VEC_CALL_INDEX()\
	/* a(intN(i,...)) */\
	/* operator(vec<int,N>) calls through operator(int...) */\
	template<typename Int, int N>\
	requires (std::is_integral_v<Int> && N <= rank)\
	constexpr decltype(auto) operator()(vec<Int, N> const & i) {\
			/* apply */\
		/*return std::apply(*this, i.s);*/\
			/* fold */\
		return [&]<auto...j>(std::index_sequence<j...>) -> decltype(auto) {\
			return (*this)(i(j)...);\
		}(std::make_index_sequence<N>{});\
	}\
	template<typename Int, int N>\
	requires (std::is_integral_v<Int> && N <= rank)\
	constexpr decltype(auto) operator()(vec<Int, N> const & i) const {\
			/* apply */\
		/*return std::apply(*this, i.s);*/\
			/* fold */\
		return [&]<auto...j>(std::index_sequence<j...>) -> decltype(auto) {\
			return (*this)(i(j)...);\
		}(std::make_index_sequence<N>{});\
	}\
	template<typename Int, int N>\
	requires (std::is_integral_v<Int> && N <= rank)\
	constexpr decltype(auto) operator()(vec<Int, N> && i) {\
			/* apply */\
		/*return std::apply(*this, i.s);*/\
			/* fold */\
		return [&]<auto...j>(std::index_sequence<j...>) -> decltype(auto) {\
			return (*this)(i(j)...);\
		}(std::make_index_sequence<N>{});\
	}\
	template<typename Int, int N>\
	requires (std::is_integral_v<Int> && N <= rank)\
	constexpr decltype(auto) operator()(vec<Int, N> && i) const {\
			/* apply */\
		/*return std::apply(*this, i.s);*/\
			/* fold */\
		return [&]<auto...j>(std::index_sequence<j...>) -> decltype(auto) {\
			return (*this)(i(j)...);\
		}(std::make_index_sequence<N>{});\
	}

#define TENSOR_ADD_SCALAR_CTOR(classname)\
	/* would be nice to have this constructor for non-tensors, non-lambdas*/\
	/* but lambas aren't invocable! */\
	/*template<typename T>*/\
	/*constexpr classname(T const & x)*/\
	/*requires (!is_tensor_v<T> && !std::is_invocable_v<T>)*/\
	/* so instead ... */\
	/*explicit */constexpr classname(Scalar const & x) {\
		/* hmm, for some reason using Inner{x} instead of Inner(x) is giving segfaults ... */\
			/* sequences */\
		[&]<size_t...k>(std::index_sequence<k...>) constexpr {\
			((s[k] = Inner(x)), ...);\
		}(std::make_index_sequence<localCount>{});\
			/* for-loops */\
		/*for (int k = 0; k < localCount; ++k) {\
			s[k] = Inner(x);\
		}*/\
	}\
	/*explicit */constexpr classname(Scalar && x) {\
			/* sequences */\
		[&]<size_t...k>(std::index_sequence<k...>) constexpr {\
			((s[k] = Inner(x)), ...);\
		}(std::make_index_sequence<localCount>{});\
			/* for-loops */\
		/*for (int k = 0; k < localCount; ++k) {\
			s[k] = Inner(x);\
		}*/\
	}

template<typename T> struct decay_unwrap_reference : public std::decay<std::unwrap_reference_t<T>> {};
template<typename T> using decay_unwrap_reference_t = typename decay_unwrap_reference<T>::type;

template<typename T> struct decay_unwrap_ref_decay : public std::decay<std::unwrap_ref_decay_t<T>> {};
template<typename T> using decay_unwrap_ref_decay_t = typename decay_unwrap_ref_decay<T>::type;

// vector cast operator
// TODO not sure how to write this to generalize into sym and others (or if I should try to?)
// explicit 'this->s' so subclasses can use this macro (like quat)
#define TENSOR_ADD_CTOR_FOR_GENERIC_TENSORS(classname)\
\
	template<typename T, T... I>\
	/*explicit*/ constexpr classname(std::integer_sequence<T, I...>) : classname{I...} {}\
\
	template<typename U>\
	/* do I want to force matching bounds or bounds-check each read? */\
	requires (\
		is_tensor_v<U> &&\
			/* require dimensions to match */\
		/*std::is_same_v<dimseq, typename U::dimseq>*/\
			/* require rank to match */\
		rank == U::rank\
			/* require scalars to be convertible */\
		/*std::is_convertible_v<decay_unwrap_ref_decay_t<Scalar>, decay_unwrap_ref_decay_t<typename U::Scalar>>*/\
	)\
	/*explicit*//* "no known conversion" for assigning vec<T>'s from vec<reference_wrapper<T>>'s */\
	constexpr classname(U const & t) {\
		auto w = write();\
		for (auto i = w.begin(); i != w.end(); ++i) {\
			/* TODO instead an index range iterator that spans the minimum of dims of this and t */\
			if (U::validIndex(i.readIndex)) {\
				/* If we use operator()(intN<>) access working for asym ... */\
				/**i = (Scalar)t(i.readIndex);*/\
				/* ... or just replace the internal storage with std::array ... */\
				*i = std::apply(t, i.readIndex.s);\
			} else {\
				*i = Scalar();\
			}\
		}\
	}\
	template<typename U>\
	requires (\
		is_tensor_v<U> &&\
			/* require dimensions to match */\
		/*std::is_same_v<dimseq, typename U::dimseq>*/\
			/* require rank to match */\
		rank == U::rank\
			/* require scalars to be convertible */\
		/*std::is_convertible_v<decay_unwrap_ref_decay_t<Scalar>, decay_unwrap_ref_decay_t<typename U::Scalar>>*/\
	)\
	/*explicit*//* "no known conversion" */\
	constexpr classname(U && t) {\
		auto w = write();\
		for (auto i = w.begin(); i != w.end(); ++i) {\
			if (U::validIndex(i.readIndex)) {\
				*i = std::apply(t, i.readIndex.s);\
			} else {\
				*i = Scalar();\
			}\
		}\
	}


// lambda ctor
#define TENSOR_ADD_LAMBDA_CTOR(classname)\
	/* use vec<int, rank> as our lambda index: */\
	/*explicit*/ constexpr classname(std::function<Scalar(intN)> f) {\
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
	template<typename Lambda>\
	/*explicit*/ constexpr classname(Lambda lambda)\
	requires (\
		std::is_same_v<\
			Common::FunctionFromLambda<Lambda>,\
			Common::FunctionFromTupleArgs<\
				Scalar,\
				Common::tuple_rep_t<int, rank>\
			>\
		>\
	) {\
		using Func = typename Common::FunctionFromLambda<Lambda>::FuncType;\
		Func f(lambda);\
		auto w = write();\
		for (auto i = w.begin(); i != w.end(); ++i) {\
			*i = std::apply(f, i.readIndex.s);\
		}\
	}

#define TENSOR_ADD_LIST_CTOR(classname)\
	constexpr classname(std::initializer_list<Inner> l) {\
		auto src = l.begin();\
		for (int i = 0; i < localCount && src != l.end(); ++i, ++src) {\
			s[i] = *src;\
		}\
	}

#define TENSOR_ADD_VARARG_CTOR(classname)\
	/* vararg ctor */\
	/* works as long as you give up ctor({init0...}, ...) */\
	/*  and replace it all with ctor{{init0...}, ...} */\
	template<typename ... Inners>\
	requires (\
		sizeof...(Inners) > 1 &&\
		(std::is_convertible_v<Inners, Inner> && ... && (true))\
	)\
	/*explicit*/ constexpr classname(Inners const & ... xs)\
		: s({Inner(xs)...}) {\
	}\
	template<typename ... Inners>\
	requires (\
		sizeof...(Inners) > 1 &&\
		(std::is_convertible_v<Inners, Inner> && ... && (true))\
	)\
	/*explicit*/ constexpr classname(Inners && ... xs)\
		: s({Inner(xs)...}) {\
	}

#if 1
#define TENSOR_ADD_ARG_CTOR(classname)\
\
	/* single-inner (not single-scalar , not lambda, not matching-rank tensor */\
	/*explicit*/ constexpr classname(Inner const & x)\
	requires (!std::is_same_v<Inner, Scalar>)\
	: s{x} {}\
	/*explicit*/ constexpr classname(Inner && x)\
	requires (!std::is_same_v<Inner, Scalar>)\
	: s{x} {}\
\
	TENSOR_ADD_VARARG_CTOR(classname)\
\
	/* vec1 */\
	/*explicit*/ constexpr classname(Inner const & x)\
	requires (\
		localCount >= 1 &&\
			/* works */\
		!std::is_same_v<Scalar, Inner>\
			/* should be equivalent ... but getting errors at the scalar ctor */\
		/*rank > 1*/\
	) : s({x}) {}\
	/*explicit*/ constexpr classname(Inner && x)\
	requires (\
		localCount >= 1 &&\
		!std::is_same_v<Scalar, Inner>\
		/*rank > 1*/\
	) : s({x}) {}\
\
	/* vec2 */\
	/*explicit*/ constexpr classname(\
		Inner const & s0_,\
		Inner const & s1_)\
	requires (localCount >= 2)\
	: s({s0_, s1_}) {}\
	/*explicit*/ constexpr classname(\
		Inner && s0_,\
		Inner && s1_)\
	requires (localCount >= 2)\
	: s({s0_, s1_}) {}\
\
	/* vec3, sym2, asym3 */\
	/*explicit*/ constexpr classname(\
		Inner const & s0_,\
		Inner const & s1_,\
		Inner const & s2_)\
	requires (localCount >= 3)\
	: s({s0_, s1_, s2_}) {}\
	/*explicit*/ constexpr classname(\
		Inner && s0_,\
		Inner && s1_,\
		Inner && s2_)\
	requires (localCount >= 3)\
	: s({s0_, s1_, s2_}) {}\
\
	/* vec4, quat */\
	/*explicit*/ constexpr classname(\
		Inner const & s0_,\
		Inner const & s1_,\
		Inner const & s2_,\
		Inner const & s3_)\
	requires (localCount >= 4)\
	: s({s0_, s1_, s2_, s3_}) {}\
	/*explicit*/ constexpr classname(\
		Inner && s0_,\
		Inner && s1_,\
		Inner && s2_,\
		Inner && s3_)\
	requires (localCount >= 4)\
	: s({s0_, s1_, s2_, s3_}) {}\
\
	/* sym3, asym4 */\
	/*explicit*/ constexpr classname(\
		Inner const & s0_,\
		Inner const & s1_,\
		Inner const & s2_,\
		Inner const & s3_,\
		Inner const & s4_,\
		Inner const & s5_)\
	requires (localCount >= 6)\
	: s({s0_, s1_, s2_, s3_, s4_, s5_}) {}\
	/*explicit*/ constexpr classname(\
		Inner && s0_,\
		Inner && s1_,\
		Inner && s2_,\
		Inner && s3_,\
		Inner && s4_,\
		Inner && s5_)\
	requires (localCount >= 6)\
	: s({s0_, s1_, s2_, s3_, s4_, s5_}) {}\
\
	/* sym4 */\
	/*explicit*/ constexpr classname(\
		Inner const & s0_,\
		Inner const & s1_,\
		Inner const & s2_,\
		Inner const & s3_,\
		Inner const & s4_,\
		Inner const & s5_,\
		Inner const & s6_,\
		Inner const & s7_,\
		Inner const & s8_,\
		Inner const & s9_)\
	requires (localCount >= 10)\
	: s({s0_, s1_, s2_, s3_, s4_, s5_, s6_, s7_, s8_, s9_}) {}\
	/*explicit*/ constexpr classname(\
		Inner && s0_,\
		Inner && s1_,\
		Inner && s2_,\
		Inner && s3_,\
		Inner && s4_,\
		Inner && s5_,\
		Inner && s6_,\
		Inner && s7_,\
		Inner && s8_,\
		Inner && s9_)\
	requires (localCount >= 10)\
	: s({s0_, s1_, s2_, s3_, s4_, s5_, s6_, s7_, s8_, s9_}) {}
#endif

#define TENSOR_ADD_CTORS(classname)\
	TENSOR_ADD_SCALAR_CTOR(classname)\
	TENSOR_ADD_CTOR_FOR_GENERIC_TENSORS(classname)\
	TENSOR_ADD_LAMBDA_CTOR(classname)\
	TENSOR_ADD_LIST_CTOR(classname)\
	TENSOR_ADD_ARG_CTOR(classname)

/*
ReadIterator and WriteIterator
depends on operator()(intN) for ReadIterator
depends on getLocalReadForWriteIndex() for its getReadForWriteIndex()

TODO InnerIterator (iterator i1 first) vs OuterIterator (iterator iN first)
ReadIterator vs WriteIterator
(Based on) ReadIndexIterator vs WriteIndexIterator
*/
#define TENSOR_ADD_ITERATOR()\
\
	/* inner is not in memory order, but ... meh idk why */\
	/* outer is in memory order */\
	static constexpr bool useReadIteratorOuter = true;\
	/* read iterator */\
	/* - sized to the tensor rank (includes multiple-rank nestings) */\
	/* - range is 0's to the tensor dims() */\
	/* begin implementation for RangeIterator's Owner: */\
	template<int i> constexpr int getRangeMin() const { return 0; }\
	template<int i> constexpr int getRangeMax() const { return dim<i>; }\
	decltype(auto) getIterValue(intN const & i) { return (*this)(i); }\
	decltype(auto) getIterValue(intN const & i) const { return (*this)(i); }\
	/* end implementation for RangeIterator's Owner: */\
	template<typename ThisConst>\
	using ReadIterator = RangeIteratorInnerVsOuter<rank, !useReadIteratorOuter, ThisConst>;\
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
	/* helper functions for WriteIterator */\
	static constexpr intN getReadForWriteIndex(intW const & i) {\
		intN res;\
		res.template subset<This::localRank, 0>() = This::getLocalReadForWriteIndex(i[0]);\
		if constexpr (numNestings > 1) {\
			/*static_assert(rank - This::localRank == Inner::rank);*/\
			res.template subset<rank-This::localRank, This::localRank>() = Inner::getReadForWriteIndex(i.template subset<numNestings-1,1>());\
		}\
		return res;\
	}\
\
	template<typename ThisConst, int numNestings_>\
	struct GetByWriteIndexImpl {\
		static constexpr decltype(auto) value(ThisConst & t, intW const & index) {\
			TENSOR_INSERT_BOUNDS_CHECK(index.s[0]);\
			return Inner::getByWriteIndex(\
				t.s[index.s[0]],\
				index.template subset<numNestings-1,1>()\
			);\
		}\
	};\
	template<typename ThisConst>\
	struct GetByWriteIndexImpl<ThisConst, 1> {\
		static constexpr decltype(auto) value(ThisConst & t, intW const & index) {\
			TENSOR_INSERT_BOUNDS_CHECK(index.s[0]);\
			return t.s[index.s[0]];\
		}\
	};\
	template<typename ThisConst>\
	static constexpr decltype(auto) getByWriteIndex(ThisConst & t, intW const & index) {\
		return GetByWriteIndexImpl<ThisConst, numNestings>::value(t, index);\
	}\
\
	template<typename WriteOwnerConst>\
	struct Write {\
		using intW = vec<int, numNestings>;\
		using intR = intN;\
\
		WriteOwnerConst & owner;\
\
		Write(WriteOwnerConst & owner_) : owner(owner_) {}\
		Write(Write const & o) : owner(o.owner) {}\
		Write(Write && o) : owner(o.owner) {}\
		Write & operator=(Write const & o) { owner = o.owner; return *this; }\
		Write & operator=(Write && o) { owner = o.owner; return *this; }\
\
		/* false = not in memory order, but ... meh idk why */\
		/* true = in memory order */\
		static constexpr bool useWriteIteratorOuter = true;\
\
		/* begin implementation for RangeIterator's Owner: */\
		template<int i> constexpr int getRangeMin() const { return 0; }\
		template<int i> constexpr int getRangeMax() const { return This::template count<i>; }\
		decltype(auto) getIterValue(intW const & i) { return getByWriteIndex<WriteOwnerConst>(owner, i); }\
		decltype(auto) getIterValue(intW const & i) const { return getByWriteIndex<WriteOwnerConst>(owner, i); }\
		/* end implementation for RangeIterator's Owner: */\
\
		/* write iterator */\
		/* - sized to the # of nestings */\
		/* - range is 0's to each nesting's .localCount */\
		/* TODO can't convert this to RangeIterator without losing the readIndex property ... or how to optionally insert it into RangeIterator? */\
		template<typename WriteConst>\
		struct WriteIterator : public RangeIteratorInnerVsOuter<numNestings, !useReadIteratorOuter, WriteConst> {\
			using Super = RangeIteratorInnerVsOuter<numNestings, !useReadIteratorOuter, WriteConst>;\
			intR readIndex;\
			WriteIterator(WriteConst & owner_, intW index_ = {})\
			: Super(owner_, index_) {\
				readIndex = This::getReadForWriteIndex(Super::index);\
			}\
			WriteIterator(WriteIterator const & o) : Super(o), readIndex(o.readIndex) {}\
			WriteIterator(WriteIterator && o) : Super(o), readIndex(o.readIndex) {}\
			WriteIterator(Super const & o) : Super(o) {\
				readIndex = This::getReadForWriteIndex(Super::index);\
			}\
			WriteIterator(Super && o) : Super(o) {\
				readIndex = This::getReadForWriteIndex(Super::index);\
			}\
			WriteIterator & operator=(WriteIterator const & o) {\
				Super::operator=(o);\
				readIndex = o.readIndex;\
				return *this;\
			}\
			WriteIterator & operator=(WriteIterator && o) {\
				Super::operator=(o);\
				readIndex = o.readIndex;\
				return *this;\
			}\
			constexpr bool operator==(WriteIterator const & o) const {\
				return &this->owner == &o.owner && Super::index == o.index;\
			}\
			constexpr bool operator!=(WriteIterator const & o) const {\
				return !operator==(o);\
			}\
\
			WriteIterator & operator++() {\
				Super::operator++();\
				readIndex = This::getReadForWriteIndex(Super::index);\
				return *this;\
			}\
			WriteIterator & operator++(int) {\
				Super::operator++();\
				readIndex = This::getReadForWriteIndex(Super::index);\
				return *this;\
			}\
\
			static WriteIterator begin(WriteConst & owner) {\
				WriteIterator i = Super::begin(owner);\
				i.readIndex = This::getReadForWriteIndex(i.index);\
				return i;\
			}\
			static WriteIterator end(WriteConst & owner) {\
				return WriteIterator(Super::end(owner));\
			}\
		};\
\
		/* TODO if .write() was called by a const object, so WriteOwnerConst is 'This const' */\
		/* then should .begin() .end() be allowed?  or should they be forced to be 'This const' as well? */\
		using iterator = WriteIterator<Write>;\
		iterator begin() { return iterator::begin(*this); }\
		iterator end() { return iterator::end(*this); }\
		using const_iterator = WriteIterator<Write const>;\
		const_iterator begin() const { return const_iterator::begin(*this); }\
		const_iterator end() const { return const_iterator::end(*this); }\
		const_iterator cbegin() const { return const_iterator::begin(*this); }\
		const_iterator cend() const { return const_iterator::end(*this); }\
	};\
\
	Write<This> write() { return Write<This>(*this); }\
	/* wait, if Write<This> write() is called by a const object ... then the return type is const ... could I detect that from within Write to forward on to Write's inner class ctor? */\
	Write<This const> write() const { return Write<This const>(*this); }

// TODO unroll the loop / constexpr lambda
#define TENSOR_ADD_VALID_INDEX()\
	template<int j>\
	struct ValidIndexImpl {\
		static constexpr bool value(intN const & i) {\
			if constexpr (j == rank) {\
				return true;\
			} else {\
				if (i(j) < 0 || i(j) >= dim<j>) return false;\
				return ValidIndexImpl<j+1>::value(i);\
			}\
		}\
	};\
	static constexpr bool validIndex(typename This::intN const & i) {\
		return ValidIndexImpl<0>::value(i);\
	}

/*
member methods as wrappers to forward to Tensor namespace methods.

Bit of a hack: MOst these are written in terms of 'This'
'This' is usu my name for the same class we are currently in.
*/
#define TENSOR_ADD_MATH_MEMBER_FUNCS()\
\
	auto elemMul(This const & o) const {\
		return Tensor::elemMul(*this, o);\
	}\
	auto elemMul(This && o) && {\
		return Tensor::elemMul(std::move(*this), std::forward<This>(o));\
	}\
\
	auto matrixCompMult(This const & o) const {\
		return Tensor::matrixCompMult(*this, o);\
	}\
	auto matrixCompMult(This && o) && {\
		return Tensor::matrixCompMult(std::move(*this), std::forward<This>(o));\
	}\
\
	auto hadamard(This const & o) const {\
		return Tensor::hadamard(*this, o);\
	}\
	auto hadamard(This && o) && {\
		return Tensor::hadamard(std::move(*this), std::forward<This>(o));\
	}\
\
	template<typename B>\
	auto inner(B const & o) const {\
		return Tensor::inner(*this, o);\
	}\
	template<typename B>\
	auto inner(B && o) && {\
		return Tensor::inner(std::move(*this), std::forward<B>(o));\
	}\
\
	auto dot(This const & o) const { return Tensor::dot(*this, o); }\
	auto dot(This && o) && { return Tensor::dot(std::move(*this), std::forward<This>(o)); }\
\
	Scalar distance(This const & o) const {\
		return Tensor::distance(*this, o);\
	}\
	Scalar distance(This && o) && {\
		return Tensor::distance(std::move(*this), std::forward<This>(o));\
	}\
\
	This normalize() const {\
		return Tensor::normalize(*this);\
	}\
\
	template<typename B>\
	requires IsBinaryTensorR3xR3Op<This, B>\
	auto cross(B const & b) const {\
		return Tensor::cross(*this, b);\
	}\
	template<typename B>\
	requires IsBinaryTensorR3xR3Op<This, B>\
	auto cross(B && b) && {\
		return Tensor::cross(std::move(*this), std::forward<B>(b));\
	}\
\
	auto outer(auto const & b) const {\
		return Tensor::outer(*this, b);\
	}\
	template<typename B>\
	auto outer(B && b) && {\
		return Tensor::outer(std::move(*this), std::forward<B>(b));\
	}\
\
	template<typename B>\
	auto outerProduct(B const & o) const {\
		return Tensor::outerProduct(*this, o);\
	}\
	template<typename B>\
	auto outerProduct(B && o) && {\
		return Tensor::outerProduct(std::move(*this), std::forward<B>(o));\
	}\
\
	template<int m=0, int n=1>\
	requires (rank >= 2)\
	auto transpose() const {\
		return Tensor::transpose<m,n>(*this);\
	}\
\
	template<int m=0, int n=1>\
	requires (\
		m < rank\
		&& n < rank\
		&& This::template dim<m> == This::template dim<n>\
	) auto contract() const {\
		return Tensor::contract<m, n>(*this);\
	}\
\
	template<int m=0, int n=1>\
	auto trace() const {\
		return Tensor::trace<m,n,This>(*this);\
	}\
\
	template<int index=0, int count=1>\
	auto contractN() const {\
		return Tensor::contractN<index, count>(*this);\
	}\
\
	template<int num=1, typename T>\
	auto interior(T const & o) const {\
		return Tensor::interior<num,This,T>(*this, o);\
	}\
	template<int num=1, typename T>\
	auto interior(T && o) && {\
		return Tensor::interior<num,This,T>(std::move(*this), std::forward<T>(o));\
	}\
\
	template<int m=0>\
	auto diagonal() const {\
		return Tensor::diagonal<m>(*this);\
	}\
\
	auto makeSym() const\
	requires (isSquare) {\
		return Tensor::makeSym(*this);\
	}\
\
	auto makeAsym() const\
	requires (isSquare) {\
		return Tensor::makeAsym(*this);\
	}\
\
	auto wedge(auto const & b) const {\
		return Tensor::wedge(*this, b);\
	}\
	template<typename B>\
	auto wedge(B && b) && {\
		return Tensor::wedge(std::move(*this), std::forward<B>(b));\
	}\
\
	auto hodgeDual() const\
	requires (isSquare) {\
		return Tensor::hodgeDual(*this);\
	}\
\
	auto dual() const\
	requires (isSquare) {\
		return Tensor::dual(*this);\
	}\
\
	auto wedgeAll() const {\
		return Tensor::wedgeAll(*this);\
	}\
\
	Scalar innerExt(This const & o) const { return Tensor::innerExt(*this, o); }\
	Scalar innerExt(This && o) const { return Tensor::innerExt(*this, o); }\
\
	Scalar normExtSq() const { return Tensor::normExtSq(*this); }\
	Scalar normExt() const { return Tensor::normExt(*this); }\
\
	Scalar lenSq() const { return Tensor::lenSq(*this); }\
	Scalar normSq() const { return Tensor::normSq(*this); }\
\
	Scalar length() const { return Tensor::length(*this); }\
	Scalar norm() const { return Tensor::norm(*this); }\
\
	Scalar measure() const { return Tensor::measure(*this); }\
	Scalar measureSimplex() const { return Tensor::measureSimplex(*this); }\
\
	template<typename B>\
	requires(\
		IsBinaryTensorOpWithMatchingNeighborDims<This, B>\
		&& B::rank == 2 /* ar - 1 + br - 1 == ar <=> br == 2 */\
	) auto operator*=(B const & b) {\
		*this = *this * b;\
		return *this;\
	}\
\
	/* these are in Tensor/Inverse.h */\
\
	Scalar determinant() const\
	requires (isSquare) {\
		return (Scalar)Tensor::determinant<This>((This const &)(*this));\
	}\
\
	This inverse(Scalar const & det) const\
	requires (isSquare) {\
		return Tensor::inverse(*this, det);\
	}\
\
	This inverse() const\
	requires (isSquare) {\
		return Tensor::inverse(*this);\
	}\
\
	/* This is dependent on the LAMBDA ctor, so maybe I should put it there? */\
	/* TODO is it even necessary? why not just use the lambda ctor? All this does is abstract the indexing. */\
\
	This map(std::function<Scalar(Scalar)> f) const {\
		return This([&](intN i) -> Scalar { return f((*this)(i)); });\
	}

/*
how should I handle fully-traced tensors that return scalars?
a(i,i) a(i,i,j,j) etc ... would have resulting tensor rank 0, which means i can't create IndexAccess wrappers cuz they'd expect intN<0>'s for dereferncing, which can't work by static_assert ...
so here i've gotta vet the IndexAccess creation ...
which means duplicating some of the functionality of IndexAccess sorting out sum vs assign indexes
*/
#define TENSOR_ADD_INDEX_NOTATION_CALL()\
	template<typename ThisConst, typename IndexType, typename... IndexTypes>\
	static decltype(auto) callIndexOp(ThisConst & this_, IndexType, IndexTypes...) {\
		/* get details first because we can't make IndexAccess until we know its |AssignIndexSeq| > 0 */\
		using IndexTuple = std::tuple<IndexType, IndexTypes...>;\
		using Details = IndexAccessDetails<IndexTuple>;\
		if constexpr (Details::rank == 0) {\
			return applyTraces<typename Details::SumIndexSeq>(this_);\
		} else {\
			/* dont' instanciating this until you know IndexTuple has non-summed-indexes or else its rank = 0 and that's bad for its vector member types */\
			return IndexAccess<ThisConst, IndexTuple>(this_);\
		}\
	}\
	template<typename IndexType, typename... IndexTypes>\
	requires (Common::is_all_base_of_v<IndexBase, IndexType, IndexTypes...>)\
	decltype(auto) operator()(IndexType i1, IndexTypes... is) {\
		return callIndexOp<This>(*this, i1, is...);\
	}\
	template<typename IndexType, typename... IndexTypes>\
	requires (Common::is_all_base_of_v<IndexBase, IndexType, IndexTypes...>)\
	decltype(auto) operator()(IndexType i1, IndexTypes... is) const {\
		return callIndexOp<This const>(*this, i1, is...);\
	}

//these are all per-element assignment operators,
// so they should work fine for all tensors: vec, sym, asym, and subsequent nestings.
#define TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_ITERATOR() /* needed by TENSOR_ADD_CTORS and a lot of tensor methods */\
	TENSOR_ADD_VALID_INDEX() /* used by generic tensor ctor */\
	TENSOR_ADD_CTORS(classname) /* ctors, namely lambda ctor, needs read iterators*/ \
	TENSOR_ADD_VECTOR_OP_EQ(+=)\
	TENSOR_ADD_VECTOR_OP_EQ(-=)\
	TENSOR_ADD_VECTOR_OP_EQ(*=)\
	TENSOR_ADD_VECTOR_OP_EQ(/=)\
	TENSOR_ADD_VECTOR_OP_EQ(<<=)\
	TENSOR_ADD_VECTOR_OP_EQ(>>=)\
	TENSOR_ADD_VECTOR_OP_EQ(&=)\
	TENSOR_ADD_VECTOR_OP_EQ(|=)\
	TENSOR_ADD_VECTOR_OP_EQ(^=)\
	TENSOR_ADD_VECTOR_OP_EQ(%=)\
	TENSOR_ADD_SCALAR_OP_EQ(+=)\
	TENSOR_ADD_SCALAR_OP_EQ(-=)\
	TENSOR_ADD_SCALAR_OP_EQ(*=)\
	TENSOR_ADD_SCALAR_OP_EQ(/=)\
	TENSOR_ADD_SCALAR_OP_EQ(<<=)\
	TENSOR_ADD_SCALAR_OP_EQ(>>=)\
	TENSOR_ADD_SCALAR_OP_EQ(&=)\
	TENSOR_ADD_SCALAR_OP_EQ(|=)\
	TENSOR_ADD_SCALAR_OP_EQ(^=)\
	TENSOR_ADD_SCALAR_OP_EQ(%=)\
	TENSOR_ADD_UNARY(-)\
	TENSOR_ADD_UNARY(~)\
	TENSOR_ADD_CMP_OP()\
	TENSOR_ADD_MATH_MEMBER_FUNCS()\
	TENSOR_ADD_INDEX_NOTATION_CALL()

// vector-specific macros:


// this is the header specific to vectors.
#define TENSOR_HEADER_VECTOR_SPECIFIC()\
\
	/*this is the storage size, used for iterting across 's' */\
	/* for vectors etc it is 's' */\
	/* for (anti?)symmetric it is N*(N+1)/2 */\
	/* TODO make a 'count<int nesting>', same as dim? */\
	static constexpr int localCount = localDim;\
\
	/* this could replace the ReplaceInner template, and maybe ReplaceLocalDim and Template too */\
	/* use these with Common::tuple_apply_t<tensori, StorageTuple> to rebuild the tensor */\
	using LocalStorage = storage_vec<localDim>;

#define TENSOR_HEADER_VECTOR(classname, Inner_, localDim_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, 1)\
	TENSOR_TEMPLATE_T_I(classname)\
	TENSOR_HEADER_VECTOR_SPECIFIC()\
	TENSOR_EXPAND_TEMPLATE_TENSORR()\
	TENSOR_HEADER()\
	static constexpr std::string tensorxStr() { return std::to_string(localDim); }

#define TENSOR_VECTOR_LOCAL_READ_FOR_WRITE_INDEX()\
	/* accepts int into .s[] storage, returns intN<localRank> of how to index it using operator() */\
	static constexpr intNLocal getLocalReadForWriteIndex(int writeIndex) {\
		return intNLocal{writeIndex};\
	}

// TODO is this safe?
#define TENSOR_ADD_SUBSET_ACCESS()\
\
	/* assumes packed tensor */\
		/* compile-time offset */\
	template<int subdim, int offset = 0>\
	vec<Inner,subdim> & subset() {\
		static_assert(subdim >= 0);\
		static_assert(offset >= 0);\
		static_assert(offset + subdim <= localCount);\
		return *(vec<Inner,subdim>*)(&s[offset]);\
	}\
	template<int subdim, int offset = 0>\
	vec<Inner,subdim> const & subset() const {\
		static_assert(subdim >= 0);\
		static_assert(offset >= 0);\
		static_assert(offset + subdim <= localCount);\
		return *(vec<Inner,subdim>*)(&s[offset]);\
	}\
		/* runtime offset */\
	template<int subdim>\
	vec<Inner,subdim> & subset(int offset) {\
		static_assert(subdim >= 0);\
		static_assert(subdim <= localCount);\
		return *(vec<Inner,subdim>*)(&s[offset]);\
	}\
	template<int subdim>\
	vec<Inner,subdim> const & subset(int offset) const {\
		static_assert(subdim >= 0);\
		static_assert(subdim <= localCount);\
		return *(vec<Inner,subdim>*)(&s[offset]);\
	}

// vector.product() == the product of all components = the volume of a size reprensted by the vector
// TODO this only sums storage, not all expanded indexes.
//  I'm only using it internally for vectors .product to get the volume of a size vector
// assumes Inner operator* and operator+ exists
#define TENSOR_ADD_COMBINE_OPS()\
	Inner product() const {\
		return [&]<size_t ... k>(std::index_sequence<k...>) constexpr -> Inner {\
			return ((s[k]) * ... * (s[localCount-1]));\
		}(std::make_index_sequence<localCount-1>{});\
	}\
\
	Inner sum() const {\
		return [&]<size_t ... k>(std::index_sequence<k...>) constexpr -> Inner {\
			return ((s[k]) + ... + (s[localCount-1]));\
		}(std::make_index_sequence<localCount-1>{});\
	}

#define TENSOR_VECTOR_ADD_SUM_RESULT()\
\
	/* Tensor/scalar +- ops will sometimes have a different result tensor type */\
	/* This type will contain the *structure* of the desired sum with a scalar */\
	/* however its scalar type will still need to be replaced */\
	using ScalarSumResult = This;\
\
	/* tensor/tensor +- ops will sometimes have a dif result too */\
	/* their result is more conditional, so there's a template arg for the other type */\
	template<typename O>\
	/* can't use dims() cuz class not finished yet ... */\
	/* so just assume the math operators have the dims() == condition cuz they do */\
	requires (is_tensor_v<O> /* && dims() == O::dims()*/)\
	struct TensorSumResultImpl {\
		static constexpr auto value() {\
			if constexpr (is_zero_v<O>) {\
				return (This*)nullptr;\
			} else {\
				using I1 = ExpandMatchingLocalRank<O>;\
				return (vec<I1, localDim>*)nullptr;\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<typename O>\
	using TensorSumResult = typename TensorSumResultImpl<O>::type;

// only add these to vec and specializations
// ... so ... 'classname' is always 'vec' for this set of macros
#define TENSOR_VECTOR_CLASS_OPS(classname)\
	TENSOR_VECTOR_LOCAL_READ_FOR_WRITE_INDEX() /* needed by TENSOR_ADD_ITERATOR in TENSOR_ADD_OPS */\
	TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_VECTOR_CALL_INDEX_PRIMARY() /* operator(int) to access .s and operator[] */\
	TENSOR_ADD_RANK1_CALL_INDEX_AUX() /* operator(int, int...), operator[] */\
	TENSOR_ADD_INT_VEC_CALL_INDEX() /* operator(intN) */\
	TENSOR_ADD_SUBSET_ACCESS()\
	TENSOR_ADD_COMBINE_OPS()\
	TENSOR_VECTOR_ADD_SUM_RESULT()

// default

// this is this class.  useful for templates.  you'd be surprised.
template<typename Inner_, int localDim_>
requires (localDim_ > 0)
struct vec {
	TENSOR_HEADER_VECTOR(vec, Inner_, localDim_)
	std::array<Inner, localCount> s = {};
	constexpr vec() {}
	TENSOR_VECTOR_CLASS_OPS(vec)
};

// size == 2 specialization

template<typename Inner_>
struct vec<Inner_,2> {
	TENSOR_HEADER_VECTOR(vec, Inner_, 2)

	union {
		struct {
			Inner x, y;
		};
		struct {
			Inner s0, s1;
		};
		std::array<Inner, localCount> s = {};
	};
	constexpr vec() {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y)
	);

	TENSOR_VECTOR_CLASS_OPS(vec)

	// 2-component swizzles
#define TENSOR_VEC2_ADD_SWIZZLE2_ij(i, j)\
	auto i ## j () { return vec<std::reference_wrapper<Inner>, 2>(i, j); }\
	auto i ## j () const { return vec<std::reference_wrapper<Inner const>, 2>(i, j); }
#define TENSOR_VEC2_ADD_SWIZZLE2_i(i)\
	TENSOR_VEC2_ADD_SWIZZLE2_ij(i,x)\
	TENSOR_VEC2_ADD_SWIZZLE2_ij(i,y)
#define TENSOR3_VEC2_ADD_SWIZZLE2()\
	TENSOR_VEC2_ADD_SWIZZLE2_i(x)\
	TENSOR_VEC2_ADD_SWIZZLE2_i(y)
	TENSOR3_VEC2_ADD_SWIZZLE2()

	// 3-component swizzles
#define TENSOR_VEC2_ADD_SWIZZLE3_ijk(i, j, k)\
	auto i ## j ## k() { return vec<std::reference_wrapper<Inner>, 3>(i, j, k); }\
	auto i ## j ## k() const { return vec<std::reference_wrapper<Inner const>, 3>(i, j, k); }
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
	auto i ## j ## k ## l() { return vec<std::reference_wrapper<Inner>, 4>(i, j, k, l); }\
	auto i ## j ## k ## l() const { return vec<std::reference_wrapper<Inner const>, 4>(i, j, k, l); }
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

// size == 3 specialization

template<typename Inner_>
struct vec<Inner_,3> {
	TENSOR_HEADER_VECTOR(vec, Inner_, 3)

	union {
		struct {
			Inner x, y, z;
		};
		struct {
			Inner s0, s1, s2;
		};
		std::array<Inner, localCount> s = {};
	};
	constexpr vec() {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y),
		std::make_pair("z", &This::z)
	);

	TENSOR_VECTOR_CLASS_OPS(vec)

	// 2-component swizzles
#define TENSOR_VEC3_ADD_SWIZZLE2_ij(i, j)\
	auto i ## j () { return vec<std::reference_wrapper<Inner>, 2>(i, j); }\
	auto i ## j () const { return vec<std::reference_wrapper<Inner const>, 2>(i, j); }
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
	auto i ## j ## k() { return vec<std::reference_wrapper<Inner>, 3>(i, j, k); }\
	auto i ## j ## k() const { return vec<std::reference_wrapper<Inner const>, 3>(i, j, k); }
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
	auto i ## j ## k ## l() { return vec<std::reference_wrapper<Inner>, 4>(i, j, k, l); }\
	auto i ## j ## k ## l() const { return vec<std::reference_wrapper<Inner const>, 4>(i, j, k, l); }
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

// TODO specialization for & types -- don't initialize the s[] array (cuz in C++ you can't)
/// tho a workaround is just use std::reference<>

template<typename Inner_>
struct vec<Inner_,4> {
	TENSOR_HEADER_VECTOR(vec, Inner_, 4)

	union {
		struct {
			Inner x, y, z, w;
		};
		struct {
			Inner s0, s1, s2, s3;
		};
		std::array<Inner, localCount> s = {};
	};
	constexpr vec() {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x", &This::x),
		std::make_pair("y", &This::y),
		std::make_pair("z", &This::z),
		std::make_pair("w", &This::w)
	);

	TENSOR_VECTOR_CLASS_OPS(vec)

	// 2-component swizzles
#define TENSOR_VEC4_ADD_SWIZZLE2_ij(i, j)\
	auto i ## j () { return vec<std::reference_wrapper<Inner>, 2>(i, j); }\
	auto i ## j () const { return vec<std::reference_wrapper<Inner const>, 2>(i, j); }
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
	auto i ## j ## k() { return vec<std::reference_wrapper<Inner>, 3>(i, j, k); }\
	auto i ## j ## k() const { return vec<std::reference_wrapper<Inner const>, 3>(i, j, k); }
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
	auto i ## j ## k ## l() { return vec<std::reference_wrapper<Inner>, 4>(i, j, k, l); }\
	auto i ## j ## k ## l() const { return vec<std::reference_wrapper<Inner const>, 4>(i, j, k, l); }
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


// zero tensor of arbitrary-dim arbitrary-rank
// has no storage (tho C++ so ... just 1 byte or whatever)

#define TENSOR_HEADER_ZERO_SPECIFIC()\
\
	static constexpr int localCount = 1;\
	using LocalStorage = storage_zero<localDim>;

#define TENSOR_HEADER_ZERO(classname, Inner_, localDim_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, 1)\
	TENSOR_TEMPLATE_T_I(classname)\
	TENSOR_HEADER_ZERO_SPECIFIC()\
	TENSOR_EXPAND_TEMPLATE_TENSORR()\
	TENSOR_HEADER()

// rank-1 single-element
#define TENSOR_ZERO_LOCAL_READ_FOR_WRITE_INDEX()\
	static constexpr intNLocal getLocalReadForWriteIndex(int writeIndex) {\
		return intNLocal();\
	}\
	static constexpr int getLocalWriteForReadIndex(int i) {\
		return 0;\
	}

#define TENSOR_ADD_ZERO_CALL_INDEX_PRIMARY()\
\
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int i) {\
		return AntiSymRef<Inner>();\
	}\
\
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int i) const {\
		return AntiSymRef<Inner>();\
	}

#define TENSOR_ZERO_ADD_SUM_RESULT()\
\
	using ScalarSumResult = vec<Inner, localDim>;\
\
	template<typename O>\
	using TensorSumResult = O;

#define TENSOR_ZERO_CLASS_OPS(classname)\
	TENSOR_ZERO_LOCAL_READ_FOR_WRITE_INDEX()\
	TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_ZERO_CALL_INDEX_PRIMARY() /* operator(int) */\
	TENSOR_ADD_RANK1_CALL_INDEX_AUX() /* operator(int, int...), operator[] */\
	TENSOR_ADD_INT_VEC_CALL_INDEX() /* operator(intN) */\
	TENSOR_ZERO_ADD_SUM_RESULT()\
	static constexpr std::string tensorxStr() { return "z " + std::to_string(localDim); }

template<typename Inner_, int localDim_>
requires (localDim_ > 0)
struct zero {
	TENSOR_HEADER_ZERO(zero, Inner_, localDim_);
	std::array<Inner, localCount> s = {};
	constexpr zero() {}
	TENSOR_ZERO_CLASS_OPS(zero)
};


// rank-2 optimized storage

inline constexpr int triangleSize(int n) {
	return (n * (n + 1)) / 2;
}

inline constexpr int symIndex(int i, int j) {
	if (i > j) return symIndex(j,i);
	return i + triangleSize(j);
}

// this assumes the class has a member template<ThisConst> Accessor for its [] and (int) access
#define TENSOR_ADD_RANK2_CALL_INDEX_AUX()\
\
	/* a(i1,i2,...) := a_i1_i2_... */\
	template<typename Int1, typename Int2, typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int1> && std::is_integral_v<Int2>))\
	constexpr decltype(auto) operator()(Int1 i, Int2 j, Ints... is) { return (*this)(i,j)(is...); }\
\
	template<typename Int1, typename Int2, typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int1> && std::is_integral_v<Int2>))\
	constexpr decltype(auto) operator()(Int1 i, Int2 j, Ints... is) const { return (*this)(i,j)(is...); }\
\
	/* a(i) := a_i */\
	/* this is incomplete so it returns the Accessor */\
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int i) { return Accessor<This>(*this, i); }\
\
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int i) const { return Accessor<This const>(*this, i); }\
\
	TENSOR_ADD_BRACKET_FWD_TO_CALL()\
	TENSOR_ADD_INT_VEC_CALL_INDEX()

// symmetric matrices

/*
when sym a_(ij) sums with ...
vec: (of something) b_ij => mat c_ij: expand the next index
ident b_(ij) => sym c_ij
sym_ij => sym_ij
asym_ij => mat_ij
symR_ijk... => sum_ij-of-{vec,sym,symR} (based on # remaining indexes)}
asymR_ijk... => mat_ij-of-{vec,asym,asymR} "
*/
#define TENSOR_HEADER_SYMMETRIC_MATRIX_SPECIFIC()\
\
	static constexpr int localCount = triangleSize(localDim);\
	using LocalStorage = storage_sym<localDim>;

#define TENSOR_HEADER_SYMMETRIC_MATRIX(classname, Inner_, localDim_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, 2)\
	TENSOR_TEMPLATE_T_I(classname)\
	TENSOR_HEADER_SYMMETRIC_MATRIX_SPECIFIC()\
	TENSOR_EXPAND_TEMPLATE_TENSORR()\
	TENSOR_HEADER()

/*
Tank-2 based default indexing access.  ALl other indexing of rank-2 's relies on this.

This depends on getLocalWriteForReadIndex() which is in TENSR_*_LOCAL_READ_FOR_WRITE_INDEX.

This is used by sym , while asym uses something different since it uses the AntiSymRef accessors.
*/
#define TENSOR_ADD_SYMMETRIC_MATRIX_CALL_INDEX()\
\
	/* a(i,j) := a_ij = a_ji */\
	/* this is the direct acces */\
	/* the type should always be Inner (const) & */\
	/* symmetric has to define 1-arg operator() */\
	/* that means I can't use the default so i have to make a 2-arg recursive case */\
	template<typename Int1, typename Int2>\
	requires (std::is_integral_v<Int1> && std::is_integral_v<Int2>)\
	constexpr decltype(auto) operator()(Int1 i, Int2 j) {\
		TENSOR_INSERT_BOUNDS_CHECK(getLocalWriteForReadIndex(i,j));\
		return s[getLocalWriteForReadIndex(i,j)];\
	}\
\
	template<typename Int1, typename Int2>\
	requires (std::is_integral_v<Int1> && std::is_integral_v<Int2>)\
	constexpr decltype(auto) operator()(Int1 i, Int2 j) const {\
		TENSOR_INSERT_BOUNDS_CHECK(getLocalWriteForReadIndex(i,j));\
		return s[getLocalWriteForReadIndex(i,j)];\
	}

// currently set to upper-triangular
// swap iread 0 and 1 to get lower-triangular
// TODO use constexpr_isqrt
#define TENSOR_SYMMETRIC_MATRIX_LOCAL_READ_FOR_WRITE_INDEX()\
	static constexpr intNLocal getLocalReadForWriteIndex(int writeIndex) {\
		intNLocal iread;\
		int w = writeIndex+1;\
		for (int i = 1; w > 0; ++i) {\
			++iread(1);\
			w -= i;\
		}\
		--iread(1);\
		iread(0) = writeIndex - triangleSize(iread(1));\
		return iread;\
	}\
\
	static constexpr int getLocalWriteForReadIndex(int i, int j) {\
		return symIndex(i,j);\
	}

template<typename AccessorOwnerConst>
struct Rank2Accessor {
	//properties for Accessor as a tensor:
#if 1	// NOTICE all of this is only for the Accessor-as-tensor interoperability.  You can disable it and just don't use Accessors as tensors.
	TENSOR_THIS(Rank2Accessor)
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(
		typename AccessorOwnerConst::Inner,
		AccessorOwnerConst::localDim,
		1); // localRank==1 always cuz we hand off to the next Accessor or Owner's Inner
	//begin TENSOR_TEMPLATE_*
	template<typename T> using Template = Rank2Accessor<T>;
	template<typename NewInner> using ReplaceInner = Rank2Accessor<typename AccessorOwnerConst::template ReplaceInner<NewInner>>;
	template<int newLocalDim> using ReplaceLocalDim = Rank2Accessor<typename AccessorOwnerConst::template ReplaceLocalDim<newLocalDim>>;
	//end TENSOR_TEMPLATE_*
	//begin TENSOR_HEADER_*_SPECIFIC
	static constexpr int localCount = AccessorOwnerConst::localDim;
	using LocalStorage = storage_vec<localDim>;
	using ScalarSumResult = typename AccessorOwnerConst::ScalarSumResult;
	// treat this like (rank-1) vector storage, so use vector's TensorSumResult
	template<typename O> using TensorSumResult = typename vec<Inner, localDim>::template TensorSumResult<O>;
	//end TENSOR_HEADER_*_SPECIFIC
	TENSOR_EXPAND_TEMPLATE_TENSORR()
	TENSOR_HEADER()
#endif

	AccessorOwnerConst & owner;
	int i;

	template<typename Int>
	requires std::is_integral_v<Int>
	Rank2Accessor(AccessorOwnerConst & owner_, Int i_)
	: owner(owner_), i(i_) {}

	template<typename Int>
	requires std::is_integral_v<Int>
	Rank2Accessor(AccessorOwnerConst & owner_, vec<Int,1> i_)
	: owner(owner_), i(i_(0)) {}

	/* these should call into sym(int,int) which is always Inner (const) & */
	template<typename Int>
	requires (std::is_integral_v<Int>)
	constexpr decltype(auto) operator()(Int j) { return owner(i,j); }

	template<typename Int>
	requires (std::is_integral_v<Int>)
	constexpr decltype(auto) operator()(Int j) const { return owner(i,j); }

	/* this provides the rest of the () [] operators that will be driven by operator(int) */
	TENSOR_ADD_RANK1_CALL_INDEX_AUX()
	TENSOR_ADD_INT_VEC_CALL_INDEX()

	TENSOR_ADD_VALID_INDEX()	// this is needed for other tensors to generic-tensor-ctor using this tensor
	TENSOR_VECTOR_LOCAL_READ_FOR_WRITE_INDEX() // needed by TENSOR_ADD_ITERATOR in TENSOR_ADD_OPS
	TENSOR_ADD_ITERATOR()
	TENSOR_ADD_MATH_MEMBER_FUNCS()
};

// Accessor is used to return between-rank indexes, so if sym is rank-2 this lets you get s[i] even if the storage only represents s[i,j]
#define TENSOR_ADD_RANK2_ACCESSOR()\
	template<typename AccessorOwnerConst>\
	using Accessor = Rank2Accessor<AccessorOwnerConst>;

#define TENSOR_SYMMETRIC_MATRIX_ADD_SUM_RESULT()\
\
	using ScalarSumResult = This;\
\
	template<typename O>\
	requires (is_tensor_v<O> /*&& dims() == O::dims()*/)\
	struct TensorSumResultImpl {\
		static constexpr auto value() {\
			if constexpr (is_zero_v<O>) {\
				return (This*)nullptr;\
			} else {\
				using I2 = ExpandMatchingLocalRank<O>;\
				if constexpr (is_ident_v<O> || is_sym_v<O> || is_symR_v<O>) {\
					return (sym<I2, localDim>*)nullptr;\
				} else if constexpr (is_vec_v<O> || is_asym_v<O> || is_asymR_v<O>) {\
					return (mat<I2, localDim, localDim>*)nullptr;\
				} else {\
					/* Don't know how to add this type.  I'd use a static_assert() but those seem to even get evaluated inside unused if-constexpr blocks */\
					return nullptr;\
				}\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<typename O>\
	requires (is_tensor_v<O> && dims() == O::dims())\
	using TensorSumResult = typename TensorSumResultImpl<O>::type;

/*
for the call index operator
a 1-param is incomplete, so it should return an accessor (same as operator[])
but a 2-param is complete
and a more-than-2 will return call on the []
and therein risks applying a call to an accessor
so the accessors need nested call indexing too
*/
#define TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(classname)\
	TENSOR_ADD_RANK2_ACCESSOR()\
	TENSOR_SYMMETRIC_MATRIX_LOCAL_READ_FOR_WRITE_INDEX()\
	TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_SYMMETRIC_MATRIX_CALL_INDEX()\
	TENSOR_ADD_RANK2_CALL_INDEX_AUX()\
	TENSOR_SYMMETRIC_MATRIX_ADD_SUM_RESULT()\
	static constexpr std::string tensorxStr() { return "s " + std::to_string(localDim); }

template<typename Inner_, int localDim_>
requires (localDim_ > 0)
struct sym {
	TENSOR_HEADER_SYMMETRIC_MATRIX(sym, Inner_, localDim_)
	std::array<Inner,localCount> s = {};
	constexpr sym() {}
	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(sym)
};

template<typename Inner_>
struct sym<Inner_,2> {
	TENSOR_HEADER_SYMMETRIC_MATRIX(sym, Inner_, 2)

	union {
		struct {
			Inner x_x;
			union { Inner x_y; Inner y_x; };
			Inner y_y;
		};
		struct {
			Inner s00, s01, s11;
		};
		std::array<Inner, localCount> s = {};
	};
	constexpr sym() {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x_x", &This::x_x),
		std::make_pair("x_y", &This::x_y),
		std::make_pair("y_y", &This::y_y)
	);

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(sym)
};

template<typename Inner_>
struct sym<Inner_,3> {
	TENSOR_HEADER_SYMMETRIC_MATRIX(sym, Inner_, 3)

	union {
		struct {
			Inner x_x;
			union { Inner x_y; Inner y_x; };
			Inner y_y;
			union { Inner x_z; Inner z_x; };
			union { Inner y_z; Inner z_y; };
			Inner z_z;
		};
		struct {
			Inner s00, s01, s11, s02, s12, s22;
		};
		std::array<Inner, localCount> s = {};
	};
	constexpr sym() {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x_x", &This::x_x),
		std::make_pair("x_y", &This::x_y),
		std::make_pair("y_y", &This::y_y),
		std::make_pair("x_z", &This::x_z),
		std::make_pair("y_z", &This::y_z),
		std::make_pair("z_z", &This::z_z)
	);

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(sym)
};

template<typename Inner_>
struct sym<Inner_, 4> {
	TENSOR_HEADER_SYMMETRIC_MATRIX(sym, Inner_, 4)

	union {
		struct {
			Inner x_x;
			union { Inner x_y; Inner y_x; };
			Inner y_y;
			union { Inner x_z; Inner z_x; };
			union { Inner y_z; Inner z_y; };
			Inner z_z;
			union { Inner x_w; Inner w_x; };
			union { Inner y_w; Inner w_y; };
			union { Inner z_w; Inner w_z; };
			Inner w_w;
		};
		struct {
			Inner s00, s01, s11, s02, s12, s22, s03, s13, s23, s33;
		};
		std::array<Inner, localCount> s = {};
	};
	constexpr sym() {}

	static constexpr auto fields = std::make_tuple(
		std::make_pair("x_x", &This::x_x),
		std::make_pair("x_y", &This::x_y),
		std::make_pair("y_y", &This::y_y),
		std::make_pair("x_z", &This::x_z),
		std::make_pair("y_z", &This::y_z),
		std::make_pair("z_z", &This::z_z),
		std::make_pair("x_w", &This::x_w),
		std::make_pair("y_w", &This::y_w),
		std::make_pair("z_w", &This::z_w),
		std::make_pair("w_w", &This::w_w)
	);

	TENSOR_SYMMETRIC_MATRIX_CLASS_OPS(sym)
};

// symmetric, identity ...
// only a single storage required
// ... so it's just a wrapper
// so I guess dimension doesn't really matter.
// but meh, it works for outer product / multiply optimizations I guess?

#define TENSOR_HEADER_IDENTITY_MATRIX_SPECIFIC()\
\
	static constexpr int localCount = 1;\
	using LocalStorage = storage_ident<localDim>;

#define TENSOR_HEADER_IDENTITY_MATRIX(classname, Inner_, localDim_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, 2)\
	TENSOR_TEMPLATE_T_I(classname)\
	TENSOR_HEADER_IDENTITY_MATRIX_SPECIFIC()\
	TENSOR_EXPAND_TEMPLATE_TENSORR()\
	TENSOR_HEADER()\
	static constexpr std::string tensorxStr() { return "i " + std::to_string(localDim); }

#define TENSOR_ADD_IDENTITY_MATRIX_CALL_INDEX()\
\
	template<typename Int1, typename Int2>\
	requires (std::is_integral_v<Int1> && std::is_integral_v<Int2>)\
	constexpr decltype(auto) operator()(Int1 i, Int2 j) {\
		if (i != j) return AntiSymRef<Inner>();\
		return AntiSymRef<Inner>(std::ref(s[0]), Sign::POSITIVE);\
	}\
\
	template<typename Int1, typename Int2>\
	requires (std::is_integral_v<Int1> && std::is_integral_v<Int2>)\
	constexpr decltype(auto) operator()(Int1 i, Int2 j) const {\
		if (i != j) return AntiSymRef<Inner const>();\
		return AntiSymRef<Inner const>(std::ref(s[0]), Sign::POSITIVE);\
	}

// shouldn't even need this, cuz nobody should be calling i
#define TENSOR_IDENTITY_MATRIX_LOCAL_READ_FOR_WRITE_INDEX()\
	static constexpr intNLocal getLocalReadForWriteIndex(int writeIndex) {\
		return intNLocal();\
	}\
	static constexpr int getLocalWriteForReadIndex(int i, int j) {\
		return 0;\
	}

#define TENSOR_IDENTITY_MATRIX_ADD_SUM_RESULT()\
\
	using ScalarSumResult = sym<Inner, localDim>;\
\
	template<typename O>\
	requires (is_tensor_v<O> /*&& dims() == O::dims()*/)\
	struct TensorSumResultImpl {\
		static constexpr auto value() {\
			if constexpr (is_zero_v<O>) {\
				return (This*)nullptr;\
			} else {\
				using I2 = ExpandMatchingLocalRank<O>;\
				if constexpr (is_ident_v<O>) {\
					return (ident<I2, localDim>*)nullptr;\
				} else if constexpr (is_sym_v<O> || is_symR_v<O>) {\
					return (sym<I2, localDim>*)nullptr;\
				} else if constexpr (is_vec_v<O> || is_asym_v<O> || is_asymR_v<O>) {\
					return (mat<I2, localDim, localDim>*)nullptr;\
				} else {\
					/* Don't know how to add this type.  I'd use a static_assert() but those seem to even get evaluated inside unused if-constexpr blocks */\
					return nullptr;\
				}\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<typename O>\
	requires (is_tensor_v<O> && dims() == O::dims())\
	using TensorSumResult = typename TensorSumResultImpl<O>::type;

#define TENSOR_IDENTITY_MATRIX_CLASS_OPS(classname)\
	TENSOR_ADD_RANK2_ACCESSOR()\
	TENSOR_IDENTITY_MATRIX_LOCAL_READ_FOR_WRITE_INDEX() /* also works for Identity ... also any rank-2? */\
	TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_IDENTITY_MATRIX_CALL_INDEX()\
	TENSOR_ADD_RANK2_CALL_INDEX_AUX()\
	TENSOR_IDENTITY_MATRIX_ADD_SUM_RESULT()

//technically dim doesn't matter for storage, but it does for tensor operations
template<typename Inner_, int localDim_>
requires (localDim_ > 0)
struct ident {
	TENSOR_HEADER_IDENTITY_MATRIX(ident, Inner_, localDim_)
	std::array<Inner_, 1> s = {};
	constexpr ident() {}
	TENSOR_IDENTITY_MATRIX_CLASS_OPS(ident)
};

// TODO symmetric, scale
// TODO symmetric, diagonal

// antisymmetric matrices

#define TENSOR_HEADER_ANTISYMMETRIC_MATRIX_SPECIFIC()\
\
	static constexpr int localCount = triangleSize(localDim - 1);\
	using LocalStorage = storage_asym<localDim>;

#define TENSOR_HEADER_ANTISYMMETRIC_MATRIX(classname, Inner_, localDim_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, 2)\
	TENSOR_TEMPLATE_T_I(classname)\
	TENSOR_HEADER_ANTISYMMETRIC_MATRIX_SPECIFIC()\
	TENSOR_EXPAND_TEMPLATE_TENSORR()\
	TENSOR_HEADER()\
	static constexpr std::string tensorxStr() { return "a " + std::to_string(localDim); }

// make sure this (and the using) is set before the specific-named accessors
#define TENSOR_ADD_ANTISYMMETRIC_MATRIX_CALL_INDEX()\
\
	/* a(i,j) := a_ij = -a_ji */\
	/* this is the direct acces */\
	template<typename ThisConst>\
	static constexpr decltype(auto) callImpl(ThisConst & this_, int i, int j) {\
		using InnerConst = typename Common::constness_of<ThisConst>::template apply_to_t<Inner>;\
		if (i == j) return AntiSymRef<InnerConst>();\
		if (i > j) return callImpl<ThisConst>(this_, j, i).flip();\
		TENSOR_INSERT_BOUNDS_CHECK(symIndex(i,j-1));\
		return AntiSymRef<InnerConst>(std::ref(this_.s[symIndex(i,j-1)]), Sign::POSITIVE);\
	}\
\
	template<typename Int1, typename Int2>\
	requires (std::is_integral_v<Int1> && std::is_integral_v<Int2>)\
	constexpr decltype(auto) operator()(Int1 i, Int2 j) {\
		return callImpl<This>(*this, i, j);\
	}\
\
	template<typename Int1, typename Int2>\
	requires (std::is_integral_v<Int1> && std::is_integral_v<Int2>)\
	constexpr decltype(auto) operator()(Int1 i, Int2 j) const {\
		return callImpl<This const>(*this, i, j);\
	}

// TODO double-check this is upper-triangular
// currently set to upper-triangular, so i < j
// swap iread 0 and 1 to get lower-triangular
#define TENSOR_ANTISYMMETRIC_MATRIX_LOCAL_READ_FOR_WRITE_INDEX()\
	static constexpr intNLocal getLocalReadForWriteIndex(int writeIndex) {\
		intNLocal iread;\
		int w = writeIndex+1;\
		for (int i = 1; w > 0; ++i) {\
			++iread(1);\
			w -= i;\
		}\
		--iread(1);\
		iread(0) = writeIndex - triangleSize(iread(1));\
		++iread(1); /* for antisymmetric, skip past diagonals*/\
		return iread;\
	}

#define TENSOR_ANTISYMMETRIC_MATRIX_ADD_SUM_RESULT()\
\
	/* asym + or - a scalar is no longer asym */\
	using ScalarSumResult = tensorr<Inner, localDim, 2>;\
\
	template<typename O>\
	requires (is_tensor_v<O> /*&& dims() == O::dims()*/)\
	struct TensorSumResultImpl {\
		static constexpr auto value() {\
			if constexpr (is_zero_v<O>) {\
				return (This*)nullptr;\
			} else {\
				using I2 = ExpandMatchingLocalRank<O>;\
				if constexpr (is_asym_v<O> || is_asymR_v<O>) {\
					return (asym<I2, localDim>*)nullptr;\
				} else if constexpr (is_vec_v<O> || is_ident_v<O> || is_sym_v<O> || is_symR_v<O>) {\
					return (mat<I2, localDim, localDim>*)nullptr;\
				} else {\
					/* Don't know how to add this type.  I'd use a static_assert() but those seem to even get evaluated inside unused if-constexpr blocks */\
					return nullptr;\
				}\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<typename O>\
	requires (is_tensor_v<O> && dims() == O::dims())\
	using TensorSumResult = typename TensorSumResultImpl<O>::type;

#define TENSOR_ANTISYMMETRIC_MATRIX_CLASS_OPS(classname)\
	TENSOR_ADD_RANK2_ACCESSOR()\
	TENSOR_ANTISYMMETRIC_MATRIX_LOCAL_READ_FOR_WRITE_INDEX()\
	TENSOR_ADD_OPS(classname)\
	TENSOR_ADD_ANTISYMMETRIC_MATRIX_CALL_INDEX()\
	TENSOR_ADD_RANK2_CALL_INDEX_AUX()\
	TENSOR_ANTISYMMETRIC_MATRIX_ADD_SUM_RESULT()

/*
for asym because I need to return reference-wrappers to support the + vs - depending on the index
this means I could expose some elements as fields and some as methods to return references
but that seems inconsistent
so next thought, expose all as methods to return references
but then if I'm not using any fields then I don't need any specializations
so no specialized sizes for asym
*/
template<typename Inner_, int localDim_>
requires (localDim_ > 0)
struct asym {
	TENSOR_HEADER_ANTISYMMETRIC_MATRIX(asym, Inner_, localDim_)

	std::array<Inner, localCount> s = {};
	constexpr asym() {}

	// I figured I could do the union/struct thing like in sym, but then half would be methods that returned refs and the other half would be fields..
	// so if I just make everything methods then there is some consistancy.
	AntiSymRef<Inner		> x_x() 		requires (localDim > 0) { return (*this)(0,0); }
	AntiSymRef<Inner const	> x_x() const 	requires (localDim > 0) { return (*this)(0,0); }

	AntiSymRef<Inner		> x_y() 		requires (localDim > 1) { return (*this)(0,1); }
	AntiSymRef<Inner const	> x_y() const 	requires (localDim > 1) { return (*this)(0,1); }
	AntiSymRef<Inner		> y_x() 		requires (localDim > 1) { return (*this)(1,0); }
	AntiSymRef<Inner const	> y_x() const 	requires (localDim > 1) { return (*this)(1,0); }
	AntiSymRef<Inner		> y_y() 		requires (localDim > 1) { return (*this)(1,1); }
	AntiSymRef<Inner const	> y_y() const 	requires (localDim > 1) { return (*this)(1,1); }

	AntiSymRef<Inner		> x_z() 		requires (localDim > 2) { return (*this)(0,2); }
	AntiSymRef<Inner const	> x_z() const 	requires (localDim > 2) { return (*this)(0,2); }
	AntiSymRef<Inner		> z_x() 		requires (localDim > 2) { return (*this)(2,0); }
	AntiSymRef<Inner const	> z_x() const 	requires (localDim > 2) { return (*this)(2,0); }
	AntiSymRef<Inner		> y_z() 		requires (localDim > 2) { return (*this)(1,2); }
	AntiSymRef<Inner const	> y_z() const 	requires (localDim > 2) { return (*this)(1,2); }
	AntiSymRef<Inner		> z_y() 		requires (localDim > 2) { return (*this)(2,1); }
	AntiSymRef<Inner const	> z_y() const 	requires (localDim > 2) { return (*this)(2,1); }
	AntiSymRef<Inner		> z_z() 		requires (localDim > 2) { return (*this)(2,2); }
	AntiSymRef<Inner const	> z_z() const 	requires (localDim > 2) { return (*this)(2,2); }

	AntiSymRef<Inner		> x_w() 		requires (localDim > 3) { return (*this)(0,3); }
	AntiSymRef<Inner const	> x_w() const 	requires (localDim > 3) { return (*this)(0,3); }
	AntiSymRef<Inner		> w_x() 		requires (localDim > 3) { return (*this)(3,0); }
	AntiSymRef<Inner const	> w_x() const 	requires (localDim > 3) { return (*this)(3,0); }
	AntiSymRef<Inner		> y_w() 		requires (localDim > 3) { return (*this)(1,3); }
	AntiSymRef<Inner const	> y_w() const 	requires (localDim > 3) { return (*this)(1,3); }
	AntiSymRef<Inner		> w_y() 		requires (localDim > 3) { return (*this)(3,1); }
	AntiSymRef<Inner const	> w_y() const 	requires (localDim > 3) { return (*this)(3,1); }
	AntiSymRef<Inner		> z_w() 		requires (localDim > 3) { return (*this)(2,3); }
	AntiSymRef<Inner const	> z_w() const 	requires (localDim > 3) { return (*this)(2,3); }
	AntiSymRef<Inner		> w_z() 		requires (localDim > 3) { return (*this)(3,2); }
	AntiSymRef<Inner const	> w_z() const 	requires (localDim > 3) { return (*this)(3,2); }
	AntiSymRef<Inner		> w_w() 		requires (localDim > 3) { return (*this)(3,3); }
	AntiSymRef<Inner const	> w_w() const 	requires (localDim > 3) { return (*this)(3,3); }

	TENSOR_ANTISYMMETRIC_MATRIX_CLASS_OPS(asym)
};

// higher/arbitrary-rank tensors:

#define TENSOR_TEMPLATE_T_I_I(classname)\
\
	template<typename Inner2, int localDim2, int localRank2>\
	using Template = classname<Inner2, localDim2, localRank2>;\
\
	template<typename NewInner>\
	using ReplaceInner = Template<NewInner, localDim, localRank>;\
\
	template<int newLocalDim>\
	using ReplaceLocalDim = Template<Inner, newLocalDim, localRank>;

// helper functions used for the totally-symmetric and totally-antisymmetric tensors:

/*
higher-rank totally-symmetri (might replace sym)
https://math.stackexchange.com/a/3795166
# of elements in rank-M dim-N storage: (d + r - 1) choose r
so for r=2 we get (d+1)! / (2! (d-1)!) = d * (d + 1) / 2
*/

// totally-symmetric

#define TENSOR_HEADER_TOTALLY_SYMMETRIC_SPECIFIC()\
\
	static constexpr int localCount = consteval_symmetricSize(localDim_, localRank_);\
	using LocalStorage = storage_symR<localDim, localRank>;

// Expand index 0 of asym^N => vec ⊗ asym^(N-1)
// Expand index N-1 of asym^N => asym^(N-1) ⊗ vec
// TODO Remove of an index will Expand it first ... but we can also shorcut Remove to *always* use this type.
#define TENSOR_EXPAND_TEMPLATE_TOTALLY_SYMMETRIC()\
\
	template<int index>\
	requires (index >= 0 && index < localRank)\
	struct ExpandLocalStoragImpl {\
		static constexpr auto value() {\
			return Common::tuple_cat_t<\
				GetTupleWrappingStorageForRank<localDim, index, storage_sym, storage_symR>,\
				std::tuple<storage_vec<localDim>>,\
				GetTupleWrappingStorageForRank<localDim, localRank-index-1, storage_sym, storage_symR>\
			>();\
		}\
		using type = decltype(value());\
	};\
	template<int index>\
	requires (index >= 0 && index < localRank)\
	using ExpandLocalStorage = typename ExpandLocalStoragImpl<index>::type;

#define TENSOR_HEADER_TOTALLY_SYMMETRIC(classname, Inner_, localDim_, localRank_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, localRank_)\
	TENSOR_TEMPLATE_T_I_I(classname)\
	TENSOR_HEADER_TOTALLY_SYMMETRIC_SPECIFIC()\
	TENSOR_EXPAND_TEMPLATE_TOTALLY_SYMMETRIC()\
	TENSOR_HEADER()\
	static constexpr std::string tensorxStr() { return "S " + std::to_string(localDim) + " " + std::to_string(localRank); }

// using 'upper-triangular' i.e. i<=j<=k<=...
// right now i'm counting into this.  is there a faster way?
#define TENSOR_TOTALLY_SYMMETRIC_LOCAL_READ_FOR_WRITE_INDEX()\
	template<int j>\
	struct GetLocalReadForWriteIndexImpl {\
		static constexpr bool exec(intNLocal & iread) {\
			if constexpr (j < 0) {\
				return true;\
			} else {\
				iread[j]++;\
				if (iread[j] >= localDim) {\
					if (GetLocalReadForWriteIndexImpl<j-1>::exec(iread)) return true;\
					iread[j] = j == 0 ? 0 : iread[j-1];\
				}\
				return false;\
			}\
		}\
	};\
	static constexpr intNLocal getLocalReadForWriteIndex(int writeIndex) {\
		intNLocal iread = {};\
		for (int i = 0; i < writeIndex; ++i) {\
			if (GetLocalReadForWriteIndexImpl<localRank-1>::exec(iread)) break;\
		}\
		return iread;\
	}\
\
	/* I'm tempted to use a int... pack for targetReadIndex but the values would still be runtime so I can' exactly use a template-sort on them, so I might as well use std::sort, so it might as well be in an array ... */\
	static constexpr int getLocalWriteForReadIndex(intNLocal targetReadIndex) {\
		/* put indexes in increasing order */\
		std::sort(targetReadIndex.s.begin(), targetReadIndex.s.end());\
		/* loop until you find the element */\
		intNLocal iread;\
		for (int writeIndex = 0; writeIndex < localCount; ++writeIndex) {\
			if (iread == targetReadIndex) return writeIndex;\
			if (GetLocalReadForWriteIndexImpl<localRank-1>::exec(iread)) break;\
		}\
		/* how should I deal with bad indexes? */\
		/*throw Common::Exception() << "failed to find write index";*/\
		/* return oob range? for iteration's sake? */\
		return localCount;\
	}

// making operator()(int...) the primary, and operator()(intN<>) the secondary
// TODO forwarding of args
#define TENSOR_ADD_TOTALLY_SYMMETRIC_CALL_INDEX()\
\
	template<typename ThisConst, typename TupleSoFar, typename Int, typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int>))\
	static constexpr decltype(auto) callGtLocalRankImplFwd(ThisConst & t, TupleSoFar sofar, Int arg, Ints... is) {\
		return callGtLocalRankImpl<ThisConst>(\
			t,\
			std::tuple_cat( sofar, std::make_tuple(arg)),\
			is...);\
	}\
	template<typename ThisConst, typename TupleSoFar, typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (true))\
	static constexpr decltype(auto) callGtLocalRankImpl(ThisConst & t, TupleSoFar sofar, Ints... is) {\
		if constexpr (std::tuple_size_v<TupleSoFar> == localRank) {\
			return t.s[getLocalWriteForReadIndex(std::make_from_tuple<intNLocal>(sofar))](is...);\
		} else {\
			return callGtLocalRankImplFwd<ThisConst>(t, sofar, is...);\
		}\
	};\
\
	/* TODO any better way to write two functions at once with differing const-ness? */\
	template<typename ThisConst, typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (true))\
	static constexpr decltype(auto) callImpl(ThisConst & this_, Ints... is) {\
		constexpr int N = sizeof...(Ints);\
		if constexpr (N < localRank) {\
			return Accessor<ThisConst, N>(this_, vec<int,N>(is...));\
		} else if constexpr (N == localRank) {\
			return this_.s[getLocalWriteForReadIndex(intNLocal(is...))];\
		} else if constexpr (N > localRank) {\
			return callGtLocalRankImpl<ThisConst, std::tuple<>, Ints...>(this_, std::make_tuple(), is...);\
		}\
	}\
\
	template<typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (true))\
	constexpr decltype(auto) operator()(Ints... is) {\
		return callImpl<This>(*this, is...);\
	}\
	template<typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (true))\
	constexpr decltype(auto) operator()(Ints... is) const {\
		return callImpl<This const>(*this, is...);\
	}\
\
	TENSOR_ADD_INT_VEC_CALL_INDEX()\
	TENSOR_ADD_BRACKET_FWD_TO_CALL()

template<typename AccessorOwnerConst, int subRank>
struct RankNAccessor {
	//properties of owner, mapped here so names dont' collide with Accessor's tensor properties:
	static constexpr int ownerLocalRank = AccessorOwnerConst::localRank;
	using ownerIntN = typename AccessorOwnerConst::intN;

	//properties for Accessor as a tensor:
#if 1	// NOTICE all of this is only for the Accessor-as-tensor interoperability.  You can disable it and just don't use Accessors as tensors.
	TENSOR_THIS(RankNAccessor)
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(
		typename AccessorOwnerConst::Inner,
		AccessorOwnerConst::localDim,
		1); // localRank==1 always cuz we hand off to the next Accessor or Owner's Inner
	//begin TENSOR_TEMPLATE_*
	template<typename T, int s> using Template = RankNAccessor<T, s>;
	template<typename NewInner> using ReplaceInner = RankNAccessor<typename AccessorOwnerConst::template ReplaceInner<NewInner>, subRank>;
	template<int newLocalDim> using ReplaceLocalDim = RankNAccessor<typename AccessorOwnerConst::template ReplaceLocalDim<newLocalDim>, subRank>;
	//end TENSOR_TEMPLATE_*
	//begin TENSOR_HEADER_*_SPECIFIC
	static constexpr int localCount = AccessorOwnerConst::localDim;
	using LocalStorage = storage_vec<localDim>;
	using ScalarSumResult = typename AccessorOwnerConst::ScalarSumResult;
	// treat this like (rank-1) vector storage, so use vector's TensorSumResult
	template<typename O> using TensorSumResult = typename vec<Inner, localDim>::template TensorSumResult<O>;
	//end TENSOR_HEADER_*_SPECIFIC
	TENSOR_EXPAND_TEMPLATE_TENSORR()
	TENSOR_HEADER()
#endif

	static_assert(subRank > 0 && subRank < ownerLocalRank);
	AccessorOwnerConst & owner;
	vec<int,subRank> i;

	template<typename Int>
	requires std::is_integral_v<Int>
	RankNAccessor(AccessorOwnerConst & owner_, vec<Int,subRank> i_)
	: owner(owner_), i(i_) {}

	/* until I can think of how to split off parameter-packs at a specific length, I'll just do this in vec<int> first like below */
	/* if subRank + sizeof...(Ints) > ownerLocalRank then call-through into owner's inner */
	/* if subRank + sizeof...(Ints) == ownerLocalRank then just call owner */
	/* if subRank + sizeof...(Ints) < ownerLocalRank then produce another Accessor */
	template<typename ThisConst, int N, typename Int>
	requires (std::is_integral_v<Int>)
	constexpr decltype(auto) callImpl(ThisConst & this_, vec<Int,N> const & i2) {
		if constexpr (subRank + N < ownerLocalRank) {
			/* returns Accessor */
			vec<Int,subRank+N> fulli;
			fulli.template subset<subRank,0>() = i;
			fulli.template subset<N,subRank>() = i2;
			return RankNAccessor<AccessorOwnerConst, subRank+N>(owner, fulli);
		} else if constexpr (subRank + N == ownerLocalRank) {
			/* returns Inner, or for asym returns AntiSymRef<Inner> */
			ownerIntN fulli;
			fulli.template subset<subRank,0>() = i;
			fulli.template subset<N,subRank>() = i2;
			return this_.owner(fulli);
		} else if constexpr (subRank + N > ownerLocalRank) {
			/* returns something further than Inner */
			ownerIntN firstI;
			firstI.template subset<subRank,0>() = i;
			firstI.template subset<ownerLocalRank-subRank,subRank>() = i2.template subset<ownerLocalRank-subRank, 0>();
			vec<Int, N - (ownerLocalRank - subRank)> restI = i2.template subset<N - (ownerLocalRank-subRank), ownerLocalRank-subRank>();
			return this_.owner(firstI)(restI);
		}
	}
	// TODO can't I just get rid of the non-const version?  no need to fwd?  nope.  need both.
	template<typename Int, int N>
	requires (std::is_integral_v<Int>)
	constexpr decltype(auto) operator()(vec<Int,N> const & i2) {
		return callImpl<This, N, Int>(*this, i2);
	}
	template<typename Int, int N>
	requires (std::is_integral_v<Int>)
	constexpr decltype(auto) operator()(vec<Int,N> const & i2) const {
		return callImpl<This const, N, Int>(*this, i2);
	}

	//TODO which type should I use for the vec?  1st of the pack?
	template<typename Int, typename... Ints>
	requires ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int>))
	constexpr decltype(auto) operator()(Int i, Ints... is) {
		return (*this)(vec<Int,sizeof...(Ints)+1>{i, is...});
	}
	template<typename Int, typename... Ints>
	requires ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int>))
	constexpr decltype(auto) operator()(Int i, Ints... is) const {
		return (*this)(vec<Int,sizeof...(Ints)+1>{i, is...});
	}

	TENSOR_ADD_BRACKET_FWD_TO_CALL()

	TENSOR_ADD_VALID_INDEX()	// this is needed for other tensors to generic-tensor-ctor using this tensor
	TENSOR_VECTOR_LOCAL_READ_FOR_WRITE_INDEX() // needed by TENSOR_ADD_ITERATOR in TENSOR_ADD_OPS
	TENSOR_ADD_ITERATOR()
	TENSOR_ADD_MATH_MEMBER_FUNCS()
};

#define TENSOR_ADD_RANK_N_ACCESSOR()\
	template<typename AccessorOwnerConst, int subRank>\
	using Accessor = RankNAccessor<AccessorOwnerConst, subRank>;

#define TENSOR_TOTALLY_SYMMETRIC_ADD_SUM_RESULT()\
\
	using ScalarSumResult = symR<Inner, localDim, localRank>;\
\
	template<typename O>\
	requires (is_tensor_v<O>)\
	struct TensorSumResultImpl {\
		static constexpr auto value() {\
			if constexpr (is_zero_v<O>) {\
				return (This*)nullptr;\
			} else {\
				using I2 = ExpandMatchingLocalRank<O>;\
				if constexpr (is_symR_v<O> && O::localRank >= localRank) {\
					return (symR<I2, localDim, localRank>*)nullptr;\
				} else {\
					return (tensorr<I2, localDim, localRank>*)nullptr;\
				}\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<typename O>\
	requires (is_tensor_v<O> && std::is_same_v<dimseq, typename O::dimseq>)\
	using TensorSumResult = typename TensorSumResultImpl<O>::type;

#define TENSOR_TOTALLY_SYMMETRIC_CLASS_OPS()\
	TENSOR_ADD_RANK_N_ACCESSOR()\
	TENSOR_TOTALLY_SYMMETRIC_LOCAL_READ_FOR_WRITE_INDEX() /* needed before TENSOR_ADD_ITERATOR in TENSOR_ADD_OPS */\
	TENSOR_ADD_OPS(symR)\
	TENSOR_ADD_TOTALLY_SYMMETRIC_CALL_INDEX()\
	TENSOR_TOTALLY_SYMMETRIC_ADD_SUM_RESULT()

// TODO rank restrictions ...
// should I allow rank-2 symR's? this overlaps with sym
// should I allow rank-1 symR's? this overlaps with vec
template<typename Inner_, int localDim_, int localRank_>
requires (localDim_ > 0 && localRank_ > 2)
struct symR {
	TENSOR_HEADER_TOTALLY_SYMMETRIC(symR, Inner_, localDim_, localRank_)
	std::array<Inner, localCount> s = {};
	constexpr symR() {}
	TENSOR_TOTALLY_SYMMETRIC_CLASS_OPS()
};

// totally antisymmetric

// bubble-sorts 'i', sets 'sign' if an odd # of flips were required to sort it
//  returns 'sign' or 'ZERO' if any duplicate indexes were found (and does not finish sorting)
template<int N>
Sign antisymSortAndCountFlips(vec<int,N> & i) {
	Sign sign = Sign::POSITIVE;
	for (int k = 0; k < N-1; ++k) {
		for (int j = 0; j < N-k-1; ++j) {
			if (i[j] == i[j+1]) {
				return Sign::ZERO;
			} else if (i[j] > i[j+1]) {
				std::swap(i[j], i[j+1]);
				sign = !sign;
			}
		}
	}
	return sign;
}

#define TENSOR_HEADER_TOTALLY_ANTISYMMETRIC_SPECIFIC()\
\
	static constexpr int localCount = consteval_antisymmetricSize(localDim_, localRank_);\
	using LocalStorage = storage_asymR<localDim, localRank>;

// Expand index 0 of asym^N => vec ⊗ asym^(N-1)
// Expand index N-1 of asym^N => asym^(N-1) ⊗ vec
// TODO Remove of an index will Expand it first ... but we can also shorcut Remove to *always* use this type.
#define TENSOR_EXPAND_TEMPLATE_TOTALLY_ANTISYMMETRIC()\
\
	template<int index>\
	requires (index >= 0 && index < localRank)\
	struct ExpandLocalStoragImpl {\
		static constexpr auto value() {\
			return Common::tuple_cat_t<\
				GetTupleWrappingStorageForRank<localDim, index, storage_asym, storage_asymR>,\
				std::tuple<storage_vec<localDim>>,\
				GetTupleWrappingStorageForRank<localDim, localRank-index-1, storage_asym, storage_asymR>\
			>();\
		}\
		using type = decltype(value());\
	};\
	template<int index>\
	requires (index >= 0 && index < localRank)\
	using ExpandLocalStorage = typename ExpandLocalStoragImpl<index>::type;

#define TENSOR_HEADER_TOTALLY_ANTISYMMETRIC(classname, Inner_, localDim_, localRank_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, localDim_, localRank_)\
	TENSOR_TEMPLATE_T_I_I(classname)\
	TENSOR_HEADER_TOTALLY_ANTISYMMETRIC_SPECIFIC()\
	TENSOR_EXPAND_TEMPLATE_TOTALLY_ANTISYMMETRIC()\
	TENSOR_HEADER()\
	static constexpr std::string tensorxStr() { return "A " + std::to_string(localDim) + " " + std::to_string(localRank); }

// using 'upper-triangular' i.e. i<=j<=k<=...
// right now i'm counting into this.  is there a faster way?
#define TENSOR_TOTALLY_ANTISYMMETRIC_LOCAL_READ_FOR_WRITE_INDEX()\
	template<int j>\
	struct GetLocalReadForWriteIndexImpl {\
		static constexpr bool execInner(intNLocal & iread) {\
			if constexpr (j < 0) {\
				return true;\
			} else {\
				iread[j]++;\
				if (iread[j] >= localDim) {\
					if (GetLocalReadForWriteIndexImpl<j-1>::execInner(iread)) return true;\
					iread[j] = j == 0 ? 0 : iread[j-1];\
				}\
				return false;\
			}\
		}\
		static constexpr bool exec(intNLocal & iread) {\
			for (;;) {\
				if (execInner(iread)) return true;\
				bool skip = false;\
				for (int k = 0; k < localRank-1; ++k) {\
					if (iread[k] == iread[k+1]) {\
						skip = true;\
						break;\
					}\
				}\
				if (!skip) return false;\
			}\
		}\
	};\
	static constexpr intNLocal getLocalReadForWriteIndex(int writeIndex) {\
		auto iread = intNLocal(std::make_integer_sequence<int, localRank>{});\
		for (int i = 0; i < writeIndex; ++i) {\
			if (GetLocalReadForWriteIndexImpl<localRank-1>::exec(iread)) break;\
		}\
		return iread;\
	}\
\
	/* NOTICE this assumes targetReadIndex is already sorted */\
	static constexpr int getLocalWriteForReadIndex(intNLocal targetReadIndex) {\
		/* loop until you find the element */\
		auto iread = intNLocal(std::make_integer_sequence<int, localRank>{});\
		for (int writeIndex = 0; writeIndex < localCount; ++writeIndex) {\
			if (iread == targetReadIndex) return writeIndex;\
			if (GetLocalReadForWriteIndexImpl<localRank-1>::exec(iread)) break;\
		}\
		/* how should I deal with bad indexes? */\
		/*throw Common::Exception() << "failed to find write index";*/\
		/* return oob range? for iteration's sake? */\
		return localCount;\
	}

// TODO bubble-sort, count # of flips, use that as parity, and then if any duplicate indexes exist, use a zero reference
/*
from my symmath/tensor/LeviCivita.lua
			local indexes = {...}
			-- duplicates mean 0
			for i=1,#indexes-1 do
				for j=i+1,#indexes do
					if indexes[i] == indexes[j] then return 0 end
				end
			end
			-- bubble sort, count the flips
			local parity = 1
			for i=1,#indexes-1 do
				for j=1,#indexes-i do
					if indexes[j] > indexes[j+1] then
						indexes[j], indexes[j+1] = indexes[j+1], indexes[j]
						parity = -parity
					end
				end
			end

*/
#define TENSOR_ADD_TOTALLY_ANTISYMMETRIC_CALL_INDEX()\
\
	/* In the case of AntiSymRef, if we have a partial-indexing then the Accessor is returned but the signs aren't flipped.*/\
	/* It's not until the full indexing is made that the sign is determined in the AntiSymRef. */\
	/* This way the Accessor doesn't need to remember the AntiSymRef::how state. */\
	/* and that lets us reuse the Accessor for both sym and asym. */\
	/* So equivalently, here we don't want to bubble-sort indexes until our index size is >= our localRank */\
	template<typename Int, typename ThisConst, int N>\
	requires (std::is_integral_v<Int>)\
	static constexpr decltype(auto) callVecImpl(ThisConst & this_, vec<Int,N> i) {\
		if constexpr (N < localRank) {\
			return Accessor<ThisConst, N>(this_, i);\
		} else if constexpr (N == localRank) {\
			using InnerConst = typename Common::constness_of<ThisConst>::template apply_to_t<Inner>;\
			auto sign = antisymSortAndCountFlips(i.template subset<localRank>());\
			if (sign == Sign::ZERO) return AntiSymRef<InnerConst>();\
			return AntiSymRef<InnerConst>(this_.s[getLocalWriteForReadIndex(i)], sign);\
		} else if constexpr (N > localRank) {\
			auto sign = antisymSortAndCountFlips(i.template subset<localRank>());\
			if (sign == Sign::ZERO) {\
				using R = decltype(this_(i.template subset<localRank,0>())(i.template subset<N-localRank,localRank>()));\
				return R();\
			}\
			/* call-thru of AntiSymRef returns another AntiSymRef ... */\
			auto result = this_(i.template subset<localRank,0>())(i.template subset<N-localRank,localRank>());\
			if (sign == Sign::NEGATIVE) {\
				return result.flip();\
			}\
			return result;\
		}\
	}\
	template<typename Int, int N>\
	requires (std::is_integral_v<Int>)\
	constexpr decltype(auto) operator()(vec<Int,N> i) {\
		return callVecImpl<Int, This>(*this, i);\
	}\
	template<typename Int, int N>\
	requires (std::is_integral_v<Int>)\
	constexpr decltype(auto) operator()(vec<Int,N> i) const {\
		return callVecImpl<Int, This const>(*this, i);\
	}\
\
	template<typename Int, typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int>))\
	constexpr decltype(auto) operator()(Int i, Ints... is) {\
		return (*this)(vec<Int,sizeof...(Ints)+1>{i, is...});\
	}\
	template<typename Int, typename... Ints>\
	requires ((std::is_integral_v<Ints>) && ... && (std::is_integral_v<Int>))\
	constexpr decltype(auto) operator()(Int i, Ints... is) const {\
		return (*this)(vec<Int,sizeof...(Ints)+1>{i, is...});\
	}\
\
	TENSOR_ADD_BRACKET_FWD_TO_CALL()

#define TENSOR_TOTALLY_ANTISYMMETRIC_ADD_SUM_RESULT()\
\
	/* asymR + or - a scalar is no longer asymR */\
	using ScalarSumResult = tensorr<Inner, localDim, localRank>;\
\
	template<typename O>\
	requires (is_tensor_v<O>)\
	struct TensorSumResultImpl {\
		static constexpr auto value() {\
			if constexpr (is_zero_v<O>) {\
				return (This*)nullptr;\
			} else {\
				using I2 = ExpandMatchingLocalRank<O>;\
				if constexpr (is_asymR_v<O> && O::localRank >= localRank) {\
					return (asymR<I2, localDim, localRank>*)nullptr;\
				} else {\
					return (tensorr<I2, localDim, localRank>*)nullptr;\
				}\
			}\
		}\
		using type = typename std::remove_pointer_t<decltype(value())>;\
	};\
	template<typename O>\
	requires (is_tensor_v<O> && std::is_same_v<dimseq, typename O::dimseq>)\
	using TensorSumResult = typename TensorSumResultImpl<O>::type;

#define TENSOR_TOTALLY_ANTISYMMETRIC_CLASS_OPS()\
	TENSOR_ADD_RANK_N_ACCESSOR()\
	TENSOR_TOTALLY_ANTISYMMETRIC_LOCAL_READ_FOR_WRITE_INDEX() /* needed before TENSOR_ADD_ITERATOR in TENSOR_ADD_OPS */\
	TENSOR_ADD_OPS(asymR)\
	TENSOR_ADD_TOTALLY_ANTISYMMETRIC_CALL_INDEX()\
	TENSOR_TOTALLY_ANTISYMMETRIC_ADD_SUM_RESULT()

template<typename Inner_, int localDim_, int localRank_>
requires (localDim_ > 0 && localRank_ > 2)
struct asymR {
	TENSOR_HEADER_TOTALLY_ANTISYMMETRIC(asymR, Inner_, localDim_, localRank_)
	std::array<Inner, localCount> s = {};
	constexpr asymR() {}
	TENSOR_TOTALLY_ANTISYMMETRIC_CLASS_OPS()
};

// tensor operations

template<typename A, typename B>
requires IsBinaryTensorDiffTypeButMatchingDims<A,B>
bool operator==(A const & a, B const & b) {
	for (auto i = a.begin(); i != a.end(); ++i) {
		if (a(i.index) != b(i.index)) return false;
	}
	return true;
}

template<typename A, typename B>
requires IsBinaryTensorDiffTypeButMatchingDims<A,B>
bool operator!=(A const & a, A const & b) {
	return !operator==(a,b);
}

//  tensor/scalar sum and scalar/tensor sum

/*
result type for tensor storage and scalar operation
	vec	ident	sym	asym	symR	asymR
+	vec	sym	sym	mat	symR	tensorr
-	vec	sym	sym	mat	symR	tensorr
*	vec	ident	sym	asym	symR	asymR
/	vec	ident	sym	asym	symR	asymR

ScalarSumResult contains the result type
*/


#define TENSOR_SCALAR_MUL_OP(op)\
template<typename A, typename B>\
requires (is_tensor_v<A> && !is_tensor_v<B>)\
decltype(auto) operator op(A const & a, B const & b) {\
	using AS = typename A::Scalar;\
	using RS = decltype(AS() op B());\
	using R = typename A::template ReplaceScalar<RS>;\
	return R([&](auto... is) -> RS {\
		return a(is...) op b;\
	});\
}\
\
template<typename A, typename B>\
requires (!is_tensor_v<A> && is_tensor_v<B>)\
decltype(auto) operator op(A const & a, B const & b) {\
	using BS = typename B::Scalar;\
	using RS = decltype(A() op BS());\
	using R = typename B::template ReplaceScalar<RS>;\
	return R([&](auto... is) -> RS {\
		return a op b(is...);\
	});\
}


#define TENSOR_SCALAR_SUM_OP(op)\
template<typename A, typename B>\
requires (is_tensor_v<A> && !is_tensor_v<B>)\
decltype(auto) operator op(A const & a, B const & b) {\
	using AS = typename A::Scalar;\
	using RS = decltype(AS() op B());\
	using R = typename A::ScalarSumResult::template ReplaceScalar<RS>;\
	return R([&](auto... is) -> RS {\
		return a(is...) op b;\
	});\
}\
\
template<typename A, typename B>\
requires (!is_tensor_v<A> && is_tensor_v<B>)\
decltype(auto) operator op(A const & a, B const & b) {\
	using BS = typename B::Scalar;\
	using RS = decltype(A() op BS());\
	using R = typename B::ScalarSumResult::template ReplaceScalar<RS>;\
	return R([&](auto... is) -> RS {\
		return a op b(is...);\
	});\
}

TENSOR_SCALAR_SUM_OP(+)
TENSOR_SCALAR_SUM_OP(-)
TENSOR_SCALAR_MUL_OP(*)

//TENSOR_SCALAR_MUL_OP(/)
// would be using the same type if not for divide-by-zeros
//TENSOR_SCALAR_SUM_OP(/)
// but tensor / scalar should still preserve structure
// just not scalar / tensor
// sooo .... here it is manually:

template<typename A, typename B>
requires (is_tensor_v<A> && !is_tensor_v<B>)
decltype(auto) operator /(A const & a, B const & b) {
	using AS = typename A::Scalar;
	using RS = decltype(AS() / B());
	using R = typename A::template ReplaceScalar<RS>;
	return R([&](auto... is) -> RS {
		return a(is...) / b;
	});
}

template<typename A, typename B>
requires (!is_tensor_v<A> && is_tensor_v<B>)
decltype(auto) operator /(A const & a, B const & b) {
	using BS = typename B::Scalar;
	using RS = decltype(A() / BS());
	using R = typename B::ScalarSumResult::template ReplaceScalar<RS>;
	return R([&](auto... is) -> RS {
		return a / b(is...);
	});
}

// this is distinct because it needs the require ! ostream
#define TENSOR_SCALAR_SHIFT_OP(op)\
template<typename A, typename B>\
requires (\
	is_tensor_v<A> &&\
	!is_tensor_v<B> &&\
	!std::is_base_of_v<std::ios_base, std::decay_t<B>>\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	using AS = typename A::Scalar;\
	using RS = decltype(AS() op B());\
	using R = typename A::template ReplaceScalar<RS>;\
	return R([&](auto... is) -> RS {\
		return a(is...) op b;\
	});\
}\
\
template<typename A, typename B>\
requires (\
	!is_tensor_v<A> &&\
	!std::is_base_of_v<std::ios_base, std::decay_t<A>> &&\
	is_tensor_v<B>\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	using BS = typename B::Scalar;\
	using RS = decltype(A() op BS());\
	using R = typename B::template ReplaceScalar<RS>;\
	return R([&](auto... is) -> RS {\
		return a op b(is...);\
	});\
}



//  tensor/tensor op

#define TENSOR_TENSOR_OP(op)\
\
/* works with arbitrary storage.  so sym+asym = mat */\
/* TODO PRESERVE MATCHING STORAGE OPTIMIZATIONS */\
template<typename A, typename B>\
/*requires IsBinaryTensorDiffTypeButMatchingDims<A,B>*/\
requires (\
	IsBinaryTensorOp<A,B>\
	&& std::is_same_v<typename A::dimseq, typename B::dimseq>\
	&& !std::is_same_v<A, B> /* because that is caught next, until I get this to preserve storage opts...*/\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	using RS = decltype(typename A::Scalar() op typename B::Scalar());\
	using R = typename A::template TensorSumResult<B>::template ReplaceScalar<RS>;\
	return R(\
		[&](auto... is) -> RS {\
			return a(is...) op b(is...);\
		});\
}\
\
/* until I get down preserving the storage, lets match to like types */\
template<typename T>\
requires (is_tensor_v<T>)\
T operator op(T const & a, T const & b) {\
	return T([&](auto... is) -> typename T::Scalar {\
		return a(is...) op b(is...);\
	});\
}

TENSOR_TENSOR_OP(+)
TENSOR_TENSOR_OP(-)
TENSOR_TENSOR_OP(/)


// integral operators

// TODO should I is_integral<ScalarType> on these:
// regardless the compiler will error on that case for me

TENSOR_TENSOR_OP(<<)
TENSOR_TENSOR_OP(>>)
TENSOR_TENSOR_OP(&)
TENSOR_TENSOR_OP(|)
TENSOR_TENSOR_OP(^)
TENSOR_TENSOR_OP(%)
// I'm too lazy to decide, SUM_OP or MUL_OP ?
TENSOR_SCALAR_SHIFT_OP(<<)
TENSOR_SCALAR_SHIFT_OP(>>)
TENSOR_SCALAR_MUL_OP(&)
TENSOR_SCALAR_MUL_OP(|)
TENSOR_SCALAR_MUL_OP(^)
TENSOR_SCALAR_MUL_OP(%)
//TENSOR_UNARY_OP(~)
// should I add these?  or should I add a boolean cast?  and would the two interfere?
//TENSOR_UNARY_OP(!)
//TENSOR_TENSOR_OP(&&)
//TENSOR_TENSOR_OP(||)
//TENSOR_TERNARY_OP(?:) ... ?


// specific typed vectors

#define TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, dim1)\
using nick##dim1 = nick##N<dim1>;

#define TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, dim1, dim2)\
using nick##dim1##x##dim2 = nick##MxN<dim1,dim2>;

#define TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, dim12)\
using nick##dim12##i##dim12 = nick##NiN<dim12>;

#define TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, dim12)\
using nick##dim12##s##dim12 = nick##NsN<dim12>;

#define TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, dim12)\
using nick##dim12##a##dim12 = nick##NaN<dim12>;

#define TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, localDim, localRank, suffix)\
using nick##suffix = nick##NsR<localDim, localRank>;

#define TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, localDim, localRank, suffix)\
using nick##suffix = nick##NaR<localDim, localRank>;

#define TENSOR_ADD_NICKNAME_TYPE(nick, ctype)\
/* typed vectors */\
template<int N> using nick##N = vec<ctype, N>;\
TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, 2)\
TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, 3)\
TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, 4)\
/* typed matrices */\
template<int M, int N> using nick##MxN = mat<ctype, M, N>;\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 2, 2)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 2, 3)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 2, 4)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 3, 2)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 3, 3)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 3, 4)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 4, 2)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 4, 3)\
TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 4, 4)\
/* identity matrix */\
template<int N> using nick##NiN = ident<ctype, N>;\
TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, 2)\
TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, 3)\
TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, 4)\
/* typed symmetric matrices */\
template<int N> using nick##NsN = sym<ctype, N>;\
TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2)\
TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3)\
TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4)\
/* typed antisymmetric matrices */\
template<int N> using nick##NaN = asym<ctype, N>;\
TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2)\
TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3)\
TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4)\
/* totally symmetric tensors */\
template<int D, int R> using nick##NsR = symR<ctype, D, R>;\
TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 3, 2s2s2)\
TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 3, 3s3s3)\
TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 3, 4s4s4)\
TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 4, 2s2s2s2)\
TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 4, 3s3s3s3)\
TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 4, 4s4s4s4)\
/* totally antisymmetric tensors */\
template<int D, int R> using nick##NaR = asymR<ctype, D, R>;\
/* can't exist: TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 3, 2a2a2)*/\
TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 3, 3a3a3)\
TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 3, 4a4a4)\
/* can't exist: TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 4, 2a2a2a2)*/\
/* can't exist: TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 4, 3a3a3a3)*/\
TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 4, 4a4a4a4)


#define TENSOR_ADD_UTYPE(x)	TENSOR_ADD_NICKNAME_TYPE(u##x,unsigned x)


#define TENSOR_ADD_TYPE(x)	TENSOR_ADD_NICKNAME_TYPE(x,x)


TENSOR_ADD_TYPE(bool)
TENSOR_ADD_TYPE(char)	// TODO 'schar' for 'signed char' or just 'char' for 'signed char' even tho that's not C convention?
TENSOR_ADD_UTYPE(char)
TENSOR_ADD_TYPE(short)
TENSOR_ADD_UTYPE(short)
TENSOR_ADD_TYPE(int)
TENSOR_ADD_UTYPE(int)
TENSOR_ADD_TYPE(float)
TENSOR_ADD_TYPE(double)
TENSOR_ADD_NICKNAME_TYPE(size, size_t)
TENSOR_ADD_NICKNAME_TYPE(intptr, intptr_t)
TENSOR_ADD_NICKNAME_TYPE(uintptr, uintptr_t)
TENSOR_ADD_NICKNAME_TYPE(ldouble, long double)


// ostream
// vec does have .fields
// and I do have my default .fields ostream in Common
//  so I added an extra flag to disable it.  (cuz TODO can I make the Common ostream fields also disable itself if a specific already exists?)
// so that the .fields vec2 vec3 vec4 and the non-.fields other vecs all look the same
// ostream uses array's ostream.
template<typename T>
requires (is_tensor_v<T>)
std::ostream & operator<<(std::ostream & o, T const & t) {
	return o << t.s;
}

} // namespace Tensor

namespace std {

// tostring

template<typename T>
requires (Tensor::is_tensor_v<T>)
string to_string(T const & t) {
	return Common::objectStringFromOStream(t);
}

/*
Ok I started on this thinking I would make vec work like an std::array
but ended up just using std::array inside vec
but I still want tuple operations (for things like structure binding) so here I am again
but whereas before I was thinking of using this for sake of its .s[] storage
now I'm thinking of this as a tensor, so rank-2 std::get<> will return Accessors etc

following: https://devblogs.microsoft.com/oldnewthing/20201015-00/?p=104369
*/

template<typename T>
requires Tensor::is_tensor_v<decay_t<T>>
struct tuple_size<T> {
	static constexpr size_t value = T::localDim;
};

template<size_t i, typename T>
requires Tensor::is_tensor_v<decay_t<T>>
struct tuple_element<i, T> {
	using type = decltype(T()[i]);
};

}

namespace Tensor {

template<std::size_t i, typename T>
requires Tensor::is_tensor_v<T>
decltype(auto) get(T & p) {
	static_assert(i < T::localDim, "index out of bounds for Tensor");
	return p[i];
}

template<std::size_t i, typename T>
requires Tensor::is_tensor_v<T>
decltype(auto) get(T && p) {
	static_assert(i < T::localDim, "index out of bounds for Tensor");
	return std::forward<T>(p)[i];
}

}

// why do I have an #include at the bottom of this file?
// because at the top I have the forward-declaration to the functions in this include
// so it had better come next or you'll get link errors
// TODO how about just put the headers at the top and these at the bottom, all in-order in Tensor.h, and make that the one entry point?
#include "Tensor/Inverse.h"
#include "Tensor/Index.h"
#include "Tensor/Range.h"
#include "Tensor/Math.h"
