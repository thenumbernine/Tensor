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
	using Parent = GenericArray<Type_, size_, ScalarType_, Child>;
	
	using Type = typename Parent::Type;
	static constexpr auto dim = dim_;
	using ScalarType = typename Parent::ScalarType;
	static constexpr auto size = Parent::size;

	/*
	initialize to identity or zero?
	zero
	- this coincides with other vector ctors
	- identity would be good for scalar types, but not so much for matrix types
	*/
	GenericDenseMatrix() : Parent() {}

	//operator() index access
	
	Type &operator()(int i, int j) { return Parent::v[Child::index(i,j)]; }
	Type const &operator()(int i, int j) const { return Parent::v[Child::index(i,j)]; }
	Type &operator()(int2 const &deref) { return Parent::v[Child::index(deref(0),deref(1))]; }
	Type const &operator()(int2 const &deref) const { return Parent::v[Child::index(deref(0),deref(1))]; }

	//operator[] index access
	
	struct IndexOperatorRef {
		GenericDenseMatrix * owner = {};
		int i = {};
		IndexOperatorRef(GenericDenseMatrix * owner_, int i_) : owner(owner_), i(i_) {}
		
		Type &operator[](int j) { return owner->v[Child::index(i,j)]; }
		Type const &operator[](int j) const { return owner->v[Child::index(i,j)]; }
	};

	IndexOperatorRef &operator[](int i) { return IndexOperatorRef(this, i); }
	IndexOperatorRef const &operator[](int i) const { return IndexOperatorRef(this, i); }
	Type &operator[](int2 const &deref) { return Parent::v[Child::index(deref(0),deref(1))]; }
	Type const &operator[](int2 const &deref) const { return Parent::v[Child::index(deref(0),deref(1))]; }
};

}
