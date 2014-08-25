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

struct Index {};

template<int rank, int j, int i>
struct FindDstForSrcIndex {
	typedef Vector<int, rank> DerefType;
	typedef Vector<Index*, rank> IndexVector;
	static bool exec(DerefType &dstForSrcIndex, const IndexVector& indexes, const IndexVector& readIndexes) {
		if (readIndexes(j) == indexes(i)) {
			dstForSrcIndex(j) = i;
			return true;	//break
		}
		return false;
	}
};

template<int rank, int j>
struct FindDstForSrcOuter {
	
	typedef Vector<int, rank> DerefType;
	
	typedef Vector<Index*, rank> IndexVector;
	
	template<int i>
	using FindDstForSrcIndexRank = FindDstForSrcIndex<rank, j, i>;
	
	static bool exec(DerefType &dstForSrcIndex, const IndexVector& indexes, const IndexVector& readIndexes) {
		bool found = ForLoop<0, rank, FindDstForSrcIndexRank>::exec(
			dstForSrcIndex,
			indexes,
			readIndexes);
		
		if (!found) throw Common::Exception() << "failed to find index";
		
		return false;
	}
};

/*
rather than this matching a Tensor for index dereferencing,
this needs its index access abstracted so that binary operations can provide their own as well
*/
template<typename Tensor_>
struct IndexAccess {
	typedef Tensor_ Tensor;
	enum { rank = Tensor::rank };
	typedef Vector<Index*, rank> IndexVector;
	typedef typename Tensor::DerefType DerefType;
	Tensor *tensor;
	IndexVector indexes;	//vector ...
	
	IndexAccess(Tensor *tensor_, IndexVector indexes_) 
	: tensor(tensor_), indexes(indexes_)
	{
	}

	IndexAccess(const IndexAccess &read);
	
	template<typename TensorB>
	IndexAccess(const IndexAccess<TensorB> &read);

	IndexAccess &operator=(const IndexAccess &read) {
		return this->operator=<Tensor>(read);
	}
	
	template<int i>
	using FindDstForSrcOuterRank = FindDstForSrcOuter<rank, i>;

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
	template<typename TensorB>
	IndexAccess &operator=(const IndexAccess<TensorB> &read) {
		static_assert(TensorB::rank == rank, "tensor assignment of differing ranks");

		DerefType dstForSrcIndex;
		ForLoop<0, rank, FindDstForSrcOuterRank>::exec(dstForSrcIndex, indexes, read.indexes);
		
		//make a copy, in case we're reading from the same source
		TensorB readSource = *read.tensor;

		std::for_each(tensor->write().begin(), tensor->write().end(),
		[&](DerefType i) {
			//for the j'th index of i ...
			// ... find indexes(j) coinciding with read.indexes(k)
			// ... and put that there
			DerefType destI;
			for (int j = 0; j < rank; ++j) {
				destI(j) = i(dstForSrcIndex(j));
			}
			(*tensor)(i) = readSource(destI);
		});
		return *this;
	}
};

template<typename Tensor>
template<typename TensorB>
IndexAccess<Tensor>::IndexAccess(const IndexAccess<TensorB> &read) {
	this->operator=(read);
}

template<typename Tensor>
IndexAccess<Tensor>::IndexAccess(const IndexAccess<Tensor> &read) 
{
	if (tensor == read.tensor && indexes == read.indexes) {
		//move constructor
		tensor = read.tensor;
		indexes = read.indexes;
	} else {
		//assignment to a permutation of the same tensor type 
		this->operator=(read);
	}
}

};

