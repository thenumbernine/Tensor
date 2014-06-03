#pragma once

#include "TensorMath/GenericDenseMatrix.h"

template<typename Type, typename OwnerType>
struct AntisymmetricMatrixAccessor {
	OwnerType *owner;
	int offset;
	bool flip;

	AntisymmetricMatrixAccessor(OwnerType *owner_, int offset_, bool flip_) 
	: owner(owner_), offset(offset_), flip(flip_) {}

	AntisymmetricMatrixAccessor &operator=(const Type &value) {
		if (owner) {
			if (flip) {
				owner->v[offset] = -value;
			} else {
				owner->v[offset] = value;
			}
		}
		return *this;
	}
	
	//instead of returning this
	// we could return a tensor whose body is this ...
	//that would solve the dereference (and any other abstraction) issues ...
	operator Type&() const { 
		if (!owner) return Type();
		if (flip) {
			return -owner->v[offset];
		} else {
			return owner->v[offset];
		}
	}
};

template<typename Type, typename OwnerType>
struct AntisymmetricMatrixAccessorConst {
	const OwnerType *owner;
	int offset;
	bool flip;

	AntisymmetricMatrixAccessorConst(const OwnerType *owner_, int offset_, bool flip_) 
	: owner(owner_), offset(offset_), flip(flip_) {}

	//instead of returning this
	// we could return a tensor whose body is this ...
	//that would solve the dereference (and any other abstraction) issues ...
	operator Type() const { 
		if (!owner) return Type();
		if (flip) {
			return -owner->v[offset];
		} else {
			return owner->v[offset];
		}
	}
};

/*
GenericAntisymmetricMatrix(i,j) == -GenericAntisymmetricMatrix(j,i)
therefore GenericAntisymmetricMatrix(i,i) == 0
*/
template<typename Type_, int dim_, typename ScalarType_, typename Child>
struct GenericAntisymmetricMatrix : public GenericDenseMatrix<Type_, dim_, ScalarType_, Child, dim_ * (dim_ - 1) / 2> {
	typedef GenericDenseMatrix<Type_, dim_, ScalarType_, Child, dim_ * (dim_ - 1) / 2> Parent;
	
	typedef typename Parent::Type Type;
	enum { dim = Parent::dim };
	typedef typename Parent::ScalarType ScalarType;
	enum { size = Parent::size };
	typedef AntisymmetricMatrixAccessor<Type_, Child> Accessor;
	typedef AntisymmetricMatrixAccessorConst<Type_, Child> AccessorConst;

	GenericAntisymmetricMatrix() : Parent() {}
	GenericAntisymmetricMatrix(const Child &a) : Parent(a) {}
	
	//index access
	Accessor operator()(int i, int j) {
		if (i == j) return Accessor(NULL, 0, false);
		if (i < j) return Accessor(this, index(i,j), false);
		if (i > j) return Accessor(this, index(j,i), true);
	}
	AccessorConst operator()(int i, int j) const {
		if (i == j) return AccessorConst(NULL, 0, false);
		if (i < j) return AccessorConst(this, index(i,j), false);
		if (i > j) return AccessorConst(this, index(j,i), true);
	}

	/*
	math-index: i is the row, j is the column
	row-major: i is nested inner-most
	upper triangular: j <= i
	*/
	static int index(int i, int j) {
		if (j > i) return index(j,i);
		//i == 0: return 0
		//i == 1: return 1 + j
		//i == 2: return 1 + 2 + j
		//i == i: return i * (i+1)/2 + j
		return i * (i + 1) / 2 + j;
	}
/*
  i->
j    0 1 2
|
v 0  0 1 3 6
  1    2 4 7
  2      5 8
           9
*/

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


