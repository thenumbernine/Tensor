#pragma once

#include "Tensor/GenericDenseMatrix.h"

namespace Tensor {

/*
GenericSymmetricMatrix(i,j) == GenericSymmetricMatrix(j,i)
*/
template<typename Type_, int dim_, typename ScalarType_, typename Child>
struct GenericSymmetricMatrix : public GenericDenseMatrix<Type_, dim_, ScalarType_, Child, dim_ * (dim_ + 1) / 2> {
	using Parent = GenericDenseMatrix<Type_, dim_, ScalarType_, Child, dim_ * (dim_ + 1) / 2>;
	
	using Type = typename Parent::Type;
	static constexpr auto dim = Parent::dim;
	using ScalarType = typename Parent::ScalarType;
	static constexpr auto size = Parent::size;

	using Parent::Parent;

	/*
	math-index: i is the row, j is the column
	row-major: i is nested inner-most
	lower triangular: j <= i
	*/
	static int index(int i, int j) {
		if (j > i) return index(j,i);
		//i == 0: return 0
		//i == 1: return 1 + j
		//i == 2: return 1 + 2 + j
		//i == i: return i * (i+1)/2 + j
		return i * (i + 1) / 2 + j;
	}

	static Vector<int,2> getReadIndexForWriteIndex(int writeIndex) {
		Vector<int,2> readIndex;
		int w = writeIndex+1;
		for (int i = 1; w > 0; ++i) {
			++readIndex(0);
			w -= i;
		}
		--readIndex(0);
		readIndex(1) = writeIndex - readIndex(0) * (readIndex(0) + 1) / 2;
		return readIndex;
	}
};

}
