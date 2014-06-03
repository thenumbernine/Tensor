#pragma once

#include "Tensor/GenericArray.h"

namespace Tensor {

/*
child being the whatever curious whatever thing that returns its child
child is used for
	vector dereference index calculation
	operator return type
*/
template<typename Type_, int dim_, typename ScalarType_, typename Child, int size_>
struct GenericDenseMatrix : public GenericArray<Type_, size_, ScalarType_, Child> {
	typedef GenericArray<Type_, size_, ScalarType_, Child> Parent;
	
	typedef typename Parent::Type Type;
	enum { dim = dim_ };
	typedef typename Parent::ScalarType ScalarType;
	enum { size = Parent::size };

	/*
	initialize to identity or zero?
	zero
	- this coincides with other vector ctors
	- identity would be good for scalar types, but not so much for matrix types
	*/
	GenericDenseMatrix() : Parent() {}

	//index access
	Type &operator()(int i, int j) { return Parent::v[Child::index(i,j)]; }
	const Type &operator()(int i, int j) const { return Parent::v[Child::index(i,j)]; }
	Type &operator()(const Vector<int,2> &deref) { return Parent::v[Child::index(deref(0),deref(1))]; }
	const Type &operator()(const Vector<int,2> &deref) const { return Parent::v[Child::index(deref(0),deref(1))]; }
};

};

