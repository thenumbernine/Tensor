#pragma once

/*
object used as a placeholder when summing across indexes
usage:
Tensor<float, Upper<3>, Lower<3>> a;
Tensor<float, Upper<3>> b;
Tensor<float, Upper<3>> c;
Index i,j;
c(i) = a(i,j) * b(j) <-performs multiplication of matrix a and vector b
*/

#include "Common/Exception.h"
#include "Common/Meta.h"

namespace Tensor {

template<char ident>
struct Index {};

template<typename IndexVector, typename ReadIndexVector, int j>
struct FindDstForSrcIndex {
	template<int i>
	struct Find {
		static constexpr auto rank = std::tuple_size_v<IndexVector>;
		using DerefType = Vector<int, rank>;
		static bool exec(DerefType& dstForSrcIndex) {
			
			struct IndexesMatch {
				static bool exec(DerefType& dstForSrcIndex) {
					dstForSrcIndex(j) = i;
					return true;	//meta-for-loop break
				}
			};
			struct IndexesDontMatch {
				static bool exec(DerefType& dstForSrcIndex) {
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
		using DerefType = Vector<int, rank>;
		
		static bool exec(DerefType &dstForSrcIndex) {
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
template<typename Tensor_, typename IndexVector>
struct IndexAccess {
	using Tensor = Tensor_;
	static constexpr auto rank = Tensor::rank;
	using DerefType = typename Tensor::DerefType;
	Tensor *tensor;

	IndexAccess(Tensor *tensor_) 
	: tensor(tensor_)
	{
	}

	template<typename TensorB, typename IndexVectorB>
	IndexAccess(const IndexAccess<TensorB, IndexVectorB> &read);

	IndexAccess &operator=(const IndexAccess &read) {
		return this->operator=<Tensor>(read);
	}

	/*
	uses operations of Tensor template parameter:
	Tensor:
		tensor->write().begin(), .end()	<- write iteration
		tensor->operator()(DerefType)	<- write operator
	
	TensorB:
		readSource.operator=	<- copy operation in case source and dest are the same (how else to circumvent this?)
		readSource.operator()(DerefType)

	If read and write are the same then we want 'read' to be copied before swizzling, to prevent overwrites of transpose operations.
	If read has higher rank then a contration is required before the copy.
	
	- compare write indexes to read indexes.  
		Find all read indexes that are not in write indexes.
		Make sure they are in pairs -- complain otherwise.
	*/
	template<typename TensorB, typename IndexVectorB>
	IndexAccess &operator=(const IndexAccess<TensorB, IndexVectorB> &read) {
		static_assert(TensorB::rank == rank, "tensor assignment of differing ranks");

		//TODO this in compile time
		// that would mean compile-time dereferences
		// which would mean no longer dereferencing by int vectors but instead by compile-time parameter packs

		DerefType dstForSrcIndex;
		Common::ForLoop<0, rank, FindDstForSrcOuter<IndexVector, IndexVectorB>::template Find>::exec(dstForSrcIndex);
	
		//assign using write iterator so the result will be pushed on stack before overwriting the write tensor
		// this way we get a copy to buffer changes between read and write, in case the same tensor is used for both
		*tensor = Tensor([&](DerefType i) {
			//for the j'th index of i ...
			// ... find indexes(j) coinciding with read.indexes(k)
			// ... and put that there
			DerefType destI;
			for (int j = 0; j < rank; ++j) {
				destI(j) = i(dstForSrcIndex(j));
			}
			return (*read.tensor)(destI);
		});
		return *this;
	}
};

//tensor indexes differ...
template<typename Tensor, typename IndexVector>
template<typename TensorB, typename IndexVectorB>
IndexAccess<Tensor, IndexVector>::IndexAccess(const IndexAccess<TensorB, IndexVectorB> &read) {
	this->operator=(read);
}

/*
template<typename Type, typename... IndexesA, typename IndexVectorA, typename... IndexesB, typename IndexVectorB>
IndexAccess< 
	result tensor type:
		of type of the greatest precision of tensor a & b's types
		or maybe just take tensor a's type
		or (currently) assert the types match
		with concatenation of indexes
		with all with any symmetric/antisymmetric modifiers removed
	result indexes:
operator*(const IndexAccess<Tensor<Type, IndexesA...>, IndexVectorA>& indexAccessA, const IndexAccess<Tensor<Type, IndexesB...>, IndexVectorB>& indexAccessB) {
}
*/

}
