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

namespace Tensor {

struct Index {};

/*
rather than this matching a Tensor for index dereferencing,
this needs its index access abstracted so that binary operations can provide their own as well
*/
template<typename Tensor_>
struct IndexAccess {
	typedef Tensor_ Tensor;
	enum { rank = Tensor::rank };
	Tensor *tensor;
	Vector<Index*, rank> indexes;	//vector ...
	
	IndexAccess(Tensor *tensor_, Vector<Index*, rank> indexes_) 
	: tensor(tensor_), indexes(indexes_)
	{
	}

	IndexAccess(const IndexAccess &read);
	
	template<typename TensorB>
	IndexAccess(const IndexAccess<TensorB> &read);

	IndexAccess &operator=(const IndexAccess &read) {
		return this->operator=<Tensor>(read);
	}

	template<typename TensorB>
	IndexAccess &operator=(const IndexAccess<TensorB> &read) {
		static_assert(TensorB::rank == rank, "tensor assignment of differing ranks");

		//TODO: this at compile time
		typedef typename Tensor::DerefType DerefType;
		DerefType dstForSrcIndex;
		for (int j = 0; j < rank; ++j) {
			bool found = false;
			for (int i = 0; i < rank; ++i) {
				if (read.indexes(j) == indexes(i)) {
					dstForSrcIndex(j) = i;
					found = true;
					break;
				}
			}
			if (!found) throw Common::Exception() << "failed to find indexes!";
		}

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

template<typename TensorA, typename TensorB>
struct BinaryOperationIndexAccess {
	const TensorA& a;
	const TensorB& b;
	BinaryOperationIndexAccess(const TensorA& a_, const TensorB& b_)
	: a(a_), b(b_) {}
};

template<typename TensorA, typename TensorB>
struct MultiplyIndexAccess : public BinaryOperationIndexAccess<TensorA, TensorB> {
	typedef BinaryOperationIndexAccess<TensorA, TensorB> Super;
	using Super::Super;
};

template<typename TensorA, typename TensorB>
MultiplyIndexAccess<TensorA, TensorB> operator*(const IndexAccess<TensorA> &a, const IndexAccess<TensorB> &b) {
	Tensor::Vector<Index*, TensorA::rank + TensorB::rank> indexes;
	for (int i = 0; i < TensorA::rank; ++i) {
		indexes(i) = a.indexes(i);
	}
	for (int i = 0; i < TensorB::rank; ++i) {
		indexes(i + TensorA::rank) = b.indexes(i);
	}
	
	return IndexAccess<MultiplyIndexAccess<TensorA, TensorB>>(
		MultiplyIndexAccess<TensorA, TensorB>(a,b), indexes);
}

};

