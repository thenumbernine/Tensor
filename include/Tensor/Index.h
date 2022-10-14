#pragma once

#include "Tensor/Index.h.h"
#include "Common/Exception.h"
#include "Common/Tuple.h"	//TupleForEach
#include "Common/Meta.h"
#include <tuple>
#include <functional>

// needs to be included *after* Vector.h

namespace Tensor {


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
		static bool exec(intN& dstForSrcIndex) {
			
			struct IndexesMatch {
				static bool exec(intN& dstForSrcIndex) {
					dstForSrcIndex(j) = i;
					return true;	//meta-for-loop break
				}
			};
			struct IndexesDontMatch {
				static bool exec(intN& dstForSrcIndex) {
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
		
		static bool exec(intN & dstForSrcIndex) {
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
template<typename TensorType_, typename IndexVector>
struct IndexAccess {
	using TensorType = TensorType_;
	static constexpr auto rank = TensorType::rank;
	using intN = typename TensorType::intN;
	TensorType & tensor;

	IndexAccess(TensorType & tensor_) 
	: tensor(tensor_)
	{}

	template<typename Tensor2Type, typename IndexVector2>
	IndexAccess(IndexAccess<Tensor2Type, IndexVector2> const & read) {
		operator=(read);
	}

	template<typename Tensor2Type, typename IndexVector2>
	IndexAccess(IndexAccess<Tensor2Type, IndexVector2> && read) {
		operator=(read);
	}


	/*
	uses operations of Tensor template parameter:
	Tensor:
		tensor->write().begin(), .end()	<- write iteration
		tensor->operator()(intN)	<- write operator
	
	Tensor2Type:
		readSource.operator=	<- copy operation in case source and dest are the same (how else to circumvent this?)
		readSource.operator()(intN)

	If read and write are the same then we want 'read' to be copied before swizzling, to prevent overwrites of transpose operations.
	If read has higher rank then a contration is required before the copy.
	
	- compare write indexes to read indexes.  
		Find all read indexes that are not in write indexes.
		Make sure they are in pairs -- complain otherwise.
	*/
	template<typename Tensor2Type, typename IndexVector2>
	IndexAccess & operator=(IndexAccess<Tensor2Type, IndexVector2> const & read) {
		static_assert(Tensor2Type::rank == rank, "tensor assignment of differing ranks");

		//TODO this in compile time
		// that would mean compile-time dereferences
		// which would mean no longer dereferencing by int vectors but instead by compile-time parameter packs

		_vec<int,rank> dstForSrcIndex;
		Common::ForLoop<0, rank, FindDstForSrcOuter<IndexVector, IndexVector2>::template Find>::exec(dstForSrcIndex);
	
		//assign using write iterator so the result will be pushed on stack before overwriting the write tensor
		// this way we get a copy to buffer changes between read and write, in case the same tensor is used for both
		tensor = TensorType([&](intN i) {
			//for the j'th index of i ...
			// ... find indexes(j) coinciding with read.indexes(k)
			// ... and put that there
			intN destI;
			for (int j = 0; j < rank; ++j) {
				destI(j) = i(dstForSrcIndex(j));
			}
			return read.tensor(destI);
		});
		return *this;
	}
	template<typename Tensor2Type, typename IndexVector2>
	IndexAccess & operator=(IndexAccess<Tensor2Type, IndexVector2> && read) {
		static_assert(Tensor2Type::rank == rank, "tensor assignment of differing ranks");

		//TODO this in compile time
		// that would mean compile-time dereferences
		// which would mean no longer dereferencing by int vectors but instead by compile-time parameter packs

		_vec<int,rank> dstForSrcIndex;
		Common::ForLoop<0, rank, FindDstForSrcOuter<IndexVector, IndexVector2>::template Find>::exec(dstForSrcIndex);
	
		//assign using write iterator so the result will be pushed on stack before overwriting the write tensor
		// this way we get a copy to buffer changes between read and write, in case the same tensor is used for both
		tensor = TensorType([&](intN i) {
			//for the j'th index of i ...
			// ... find indexes(j) coinciding with read.indexes(k)
			// ... and put that there
			intN destI;
			for (int j = 0; j < rank; ++j) {
				destI(j) = i(dstForSrcIndex(j));
			}
			return read.tensor(destI);
		});
		return *this;
	}
};



/*
template<typename Type, typename... IndexesA, typename IndexVectorA, typename... IndexesB, typename IndexVector2>
IndexAccess< 
	result tensor type:
		of type of the greatest precision of tensor a & b's types
		or maybe just take tensor a's type
		or (currently) assert the types match
		with concatenation of indexes
		with all with any symmetric/antisymmetric modifiers removed
	result indexes:
operator*(const IndexAccess<Tensor<Type, IndexesA...>, IndexVectorA>& indexAccessA, const IndexAccess<Tensor<Type, IndexesB...>, IndexVector2>& indexAccessB) {
}
*/

template<typename T, typename Is>
std::ostream & operator<<(std::ostream & o, IndexAccess<T,Is> const & ti) {
	o << "[" << ti.tensor << "_";
	Common::TupleForEach(Is(), [&o](auto x, size_t i) constexpr -> bool {
		o << x;
		return false;
	});
	o << "]";
	return o;
}

}
