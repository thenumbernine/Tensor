#pragma once

#include "Tensor/GenericArray.h"
#include <cmath>	//sqrt
#include <functional>

namespace Tensor {

/*
adds int-based indexing and vector-vector ops to the array ops
*/
template<typename Type_, int size_, typename ScalarType_, typename Child>
struct GenericVector : public GenericArray<Type_, size_, ScalarType_, Child> {
	typedef GenericArray<Type_, size_, ScalarType_, Child> Parent;

	typedef typename Parent::Type Type;
	enum { size = Parent::size };
	typedef typename Parent::ScalarType ScalarType;

	//inherited constructors
	//using Parent::Parent;
	GenericVector() : Parent() {}
	GenericVector(const Child &a) : Parent(a) {}
	GenericVector(const Type &x) : Parent(x) {}

	GenericVector(std::function<Type(int)> &f) {
		for (int i = 0; i < size; ++i) {
			Parent::v[i] = f(i);
		}
	}

	//specific dimension initializers
	GenericVector(const Type &x, const Type &y) {
		static_assert(size >= 2, "vector not big enough");
		Parent::v[0] = x;
		Parent::v[1] = y;
		for (int i = 2; i < size; ++i) Parent::v[i] = Type();
	}

	GenericVector(const Type &x, const Type &y, const Type &z) {
		static_assert(size >= 3, "vector not big enough");
		Parent::v[0] = x;
		Parent::v[1] = y;
		Parent::v[2] = z;
		for (int i = 3; i < size; ++i) Parent::v[i] = Type();
	}

	GenericVector(const Type &x, const Type &y, const Type &z, const Type &w) {
		static_assert(size >= 4, "vector not big enough");
		Parent::v[0] = x;
		Parent::v[1] = y;
		Parent::v[2] = z;
		Parent::v[3] = w;
		for (int i = 4; i < size; ++i) Parent::v[i] = Type();
	}

	//index access
	Type &operator()(int i) { return Parent::v[i]; }
	const Type &operator()(int i) const { return Parent::v[i]; }

	//product of elements / flat-space volume operator
	Type volume() const {
		Type vol = Type(1);
		for (int i = 0; i < size; ++i) {
			vol *= Parent::v[i];
		}
		return vol;
	}

	//inner product / flat-space dot product
	static Type dot(const Child &a, const Child &b) {
		Type d = Type(0);
		for (int i = 0; i < size; ++i) {
			d += a.v[i] * b.v[i];
		}
		return d;
	}

	static double length(const Child &a) {
		Type lengthSquared = dot(a,a);
		//or provide your own sqrt and return Type?
		return sqrt((double)lengthSquared);
	}

	//vector/vector operations
	//i figure matrix doesn't want * and / to be matrix/matrix per-element
	//trying to copy GLSL as much as I can here

	Child operator*(const Child &b) const {
		const GenericVector &a = *this;
		Child c;
		for (int i = 0; i < size; ++i) {
			c.v[i] = a.v[i] * b.v[i];
		}
		return c;
	}

	Child operator*(const ScalarType &b) const { return Parent::operator*(b); }
	
	Child operator/(const Child &b) const {
		const GenericVector &a = *this;
		Child c;
		for (int i = 0; i < size; ++i) {
			c.v[i] = a.v[i] / b.v[i];
		}
		return c;
	}
	
	Child operator/(const ScalarType &b) const { return Parent::operator/(b); }
};

};

