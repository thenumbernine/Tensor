#pragma once

#include "Tensor/Index.h.h"
#include "Common/Exception.h"
#include "Common/Tuple.h"	//TupleForEach
#include "Common/Meta.h"
#include <tuple>
#include <functional>	// plus minus binary_operator etc

// needs to be included *after* Vector.h

namespace Tensor {

template<typename T>
constexpr bool is_IndexExpr_v = requires(T const & t) { &T::isIndexExprFlag; };

template<typename A, typename B>
concept IsMatchingRankExpr = 
	is_IndexExpr_v<A>
	&& is_IndexExpr_v<B>
	&& A::rank == B::rank;

template<char ch>
std::ostream & operator<<(std::ostream & o, Index<ch> const & i) {
	return o << ch;
}

template<typename IndexVector, typename ReadIndexVector, int j>
struct FindDstForSrcIndex {
	template<int i>
	struct Find {
		static constexpr auto rank = std::tuple_size_v<IndexVector>;
		using intN = _vec<int,rank>;
		static constexpr bool exec(intN& dstForSrcIndex) {
			
			struct IndexesMatch {
				static constexpr bool exec(intN& dstForSrcIndex) {
					dstForSrcIndex(j) = i;
					return true;	//meta-for-loop break
				}
			};
			struct IndexesDontMatch {
				static constexpr bool exec(intN& dstForSrcIndex) {
					return false;
				}
			};
			
			return std::conditional_t<
				std::is_same_v<
					std::tuple_element_t<j, ReadIndexVector>,
					std::tuple_element_t<i, IndexVector>
				>,
				IndexesMatch,
				IndexesDontMatch
			>::exec(dstForSrcIndex);
		}
	};
};

template<typename IndexVector, typename ReadIndexVector>
struct FindDstForSrcOuter {
	template<int j>
	struct Find {
		static constexpr auto rank = std::tuple_size_v<IndexVector>;
		using intN = _vec<int,rank>;
		
		static constexpr bool exec(intN & dstForSrcIndex) {
			bool found = Common::ForLoop<0, rank, FindDstForSrcIndex<IndexVector, ReadIndexVector, j>::template Find>::exec(dstForSrcIndex);
			if (!found) throw Common::Exception() << "failed to find index";
			return false;
		}
	};
};

/*
rather than this matching a Tensor for index dereferencing,
this needs its index access abstracted so that binary operations can provide their own as well
*/
template<typename TensorType_, typename IndexVector_>
struct IndexAccess {
	using This = IndexAccess;
	using TensorType = TensorType_;
	using IndexVector = IndexVector_;	// std::tuple<Index<char>... >
	using intN = typename TensorType::intN;
	using Scalar = typename TensorType::Scalar;
	static constexpr bool isIndexExprFlag = {};
	static constexpr auto rank = TensorType::rank;
	static constexpr auto dims() { return TensorType::dims(); }
	TensorType & t;

	IndexAccess(TensorType & t_) : t(t_) {}

	template<typename B>
	requires is_IndexExpr_v<B>
	IndexAccess(B const & read) {
		assign<B>(read);
	}
	template<typename B>
	requires is_IndexExpr_v<B>
	IndexAccess(B && read) {
		assign<B>(read);
	}

	template<typename B>
	IndexAccess & operator=(B const & read) {
		assign<B>(read);
		return *this;
	}
	template<typename B>
	IndexAccess & operator=(B && read) {
		assign<B>(std::forward<B>(read));
		return *this;
	}

	/*
	uses operations of Tensor template parameter:
	Tensor:
		t->write().begin(), .end()	<- write iteration
		t->operator()(intN)	<- write operator
	
	Tensor2Type:
		readSource.operator=	<- copy operation in case source and dest are the same (how else to circumvent this?)
		readSource.operator()(intN)

	If read and write are the same then we want 'read' to be copied before swizzling, to prevent overwrites of transpose operations.
	If read has higher rank then a contration is required before the copy.
	
	- compare write indexes to read indexes.
		Find all read indexes that are not in write indexes.
		Make sure they are in pairs -- complain otherwise.
	*/
	template<typename B>//typename Tensor2Type, typename IndexVector2>
	requires IsMatchingRankExpr<This, B>
	void assign(B const & src) {
		// TODO I could assert dim match too, or I can bounds check the src reads

		//assign using write iterator so the result will be pushed on stack before overwriting the write tensor
		// this way we get a copy to buffer changes between read and write, in case the same tensor is used for both
#if 0
		// same as TENSOR_ADD_CTOR_FOR_GENERIC_TENSORS
		// downside is it can invalidate itself if you're reading and writing to the same tensor
		auto w = t.write();
		for (auto i = w.begin(); i != w.end(); ++i) {
			*i = src.template read<IndexVector>(i.readIndex);
		};
#else	// requires an extra object on the stack but can read and write to the same tensor
		t = TensorType([&](intN i) -> Scalar {
			return src.template read<IndexVector>(i);
		});
#endif
	}

	template<typename DstIndexVector>
	static constexpr intN getIndex(intN const & i) {
		//TODO this in compile time
		// that would mean compile-time dereferences
		// which would mean no longer dereferencing by int vectors but instead by compile-time parameter packs
		_vec<int,rank> dstForSrcIndex;
		Common::ForLoop<
			0,
			rank,
			FindDstForSrcOuter<DstIndexVector, IndexVector>::template Find
		>::exec(dstForSrcIndex);
	
		//for the j'th index of i ...
		// ... find indexes(j) coinciding with read.indexes(k)
		// ... and put that there
		intN destI;
		for (int j = 0; j < rank; ++j) {
			destI(j) = i(dstForSrcIndex(j));
		}
		return destI;
	}

	// if we're just wrapping a tensor ... the read operation is just a tensor access
	// TODO decltype(auto) 
	//  to do that, constexpr the validIndex
	//  to do that, i should be constexpr
	template<typename DstIndexVector>
	constexpr Scalar read(intN const & i) const {
		auto dstI = getIndex<DstIndexVector>(i);
		if (TensorType::validIndex(dstI)) {
			return t(dstI);
		} else {
			return Scalar();
		}
	}
};

template<typename T, typename Is>
std::ostream & operator<<(std::ostream & o, IndexAccess<T,Is> const & ti) {
	o << "[" << ti.t << "_";
	Common::TupleForEach(Is(), [&o](auto x, size_t i) constexpr -> bool {
		o << x;
		return false;
	});
	o << "]";
	return o;
}

// tensor + tensor

template<typename A, typename B, template<typename> typename op>
requires IsMatchingRankExpr<A, B>
struct TensorTensorExpr {
	using intN = typename A::intN;
	using Scalar = typename A::Scalar; // TODO which Scalar to use?
	static constexpr bool isIndexExprFlag = {};
	static constexpr auto rank = A::rank;
	A const & a;
	B const & b;
	
	TensorTensorExpr(A const & a_, B const & b_) : a(a_), b(b_) {}

	template<typename DstIndexVector>
	constexpr Scalar read(intN const & i) const {
		return op<Scalar>()(
			a.template read<DstIndexVector>(i),
			b.template read<DstIndexVector>(i));
	}
};

// TODO ensure A's dims == B's dims?
// but that means recalculating dims and rank for IndexAccess and its expression-trees based on its 
// and direct assignment doesn't assert this -- instead it bounds-checks ... soo ...
//static_assert(A::TensorType::dims() == B::TensorType::dims());
template<typename A, typename B>
requires IsMatchingRankExpr<A, B>
decltype(auto) operator+(A const & a, B const & b) {
	return TensorTensorExpr<A, B, std::plus>(a,b);
}

// tensor + scalar

template<typename T, template<typename> typename op>
requires is_IndexExpr_v<T>
struct TensorScalarExpr {
	using intN = typename T::intN;
	using Scalar = typename T::Scalar; // TODO which Scalar to use?
	static constexpr bool isIndexExprFlag = {};
	static constexpr auto rank = T::rank;
	T const & a;
	Scalar const & b;
	
	TensorScalarExpr(T const & a_, Scalar const & b_) : a(a_), b(b_) {}

	template<typename DstIndexVector>
	constexpr Scalar read(intN const & i) const {
		return op<Scalar>()(a.template read<DstIndexVector>(i), b);
	}
};

template<typename T>
requires is_IndexExpr_v<T>
decltype(auto) operator+(T const & a, typename T::Scalar const & b) {
	return TensorScalarExpr<T, std::plus>(a,b);
}

// scalar + tensor

template<typename T, template<typename> typename op>
requires is_IndexExpr_v<T>
struct ScalarTensorExpr {
	using intN = typename T::intN;
	using Scalar = typename T::Scalar; // TODO which Scalar to use?
	static constexpr bool isIndexExprFlag = {};
	static constexpr auto rank = T::rank;
	Scalar const & a;
	T const & b;
	
	ScalarTensorExpr(Scalar const & a_, T const & b_) : a(a_), b(b_) {}

	template<typename DstIndexVector>
	constexpr Scalar read(intN const & i) const {
		return op<Scalar>()(a, b.template read<DstIndexVector>(i));
	}
};
template<typename T>
requires is_IndexExpr_v<T>
decltype(auto) operator+(typename T::Scalar const & a, T const & b) {
	return ScalarTensorExpr<T, std::plus>(a,b);
}

}
