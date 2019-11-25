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
	using This = GenericVector;
	using Super = GenericArray<Type_, size_, ScalarType_, Child>;

	using Type = typename Super::Type;
	static constexpr auto size = Super::size;
	using ScalarType = typename Super::ScalarType;

	//inherited constructors
	//using Super::Super;
	GenericVector() : Super() {}
	GenericVector(const Child &a) : Super(a) {}
	GenericVector(const Type &x) : Super(x) {}

	GenericVector(std::function<Type(int)> f) {
		for (int i = 0; i < size; ++i) {
			Super::v[i] = f(i);
		}
	}

	//specific dimension initializers
	GenericVector(const Type &x, const Type &y) {
		static_assert(size >= 2, "vector not big enough");
		Super::v[0] = x;
		Super::v[1] = y;
		for (int i = 2; i < size; ++i) Super::v[i] = Type();
	}

	GenericVector(const Type &x, const Type &y, const Type &z) {
		static_assert(size >= 3, "vector not big enough");
		Super::v[0] = x;
		Super::v[1] = y;
		Super::v[2] = z;
		for (int i = 3; i < size; ++i) Super::v[i] = Type();
	}

	GenericVector(const Type &x, const Type &y, const Type &z, const Type &w) {
		static_assert(size >= 4, "vector not big enough");
		Super::v[0] = x;
		Super::v[1] = y;
		Super::v[2] = z;
		Super::v[3] = w;
		for (int i = 4; i < size; ++i) Super::v[i] = Type();
	}

	//index access
	Type &operator()(int i) { return Super::v[i]; }
	const Type &operator()(int i) const { return Super::v[i]; }

	//product of elements / flat-space volume operator
	Type volume() const {
		Type vol = Type(1);
		for (int i = 0; i < size; ++i) {
			vol *= Super::v[i];
		}
		return vol;
	}

	//inner product / flat-space dot product
	static Type dot(const Child &a, const Child &b) {
		Type d = {};
		for (int i = 0; i < size; ++i) {
			d += a.v[i] * b.v[i];
		}
		return d;
	}

	Type lenSq() const {
		return dot(crtp_cast<Child>(*this), crtp_cast<Child>(*this));
	}

	Type length() const {
		return (Type)sqrt(lenSq());
	}

	Child unit(Type eps = 1e-9) const {
		Type len = length();
		if (len > eps) {
			return Child( crtp_cast<Child>(*this) ) * (Type(1) / len);
		} else {
			return Child();	//return zero or unit in some arbitrary direction?
		}
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

	Child operator*(const ScalarType &b) const { return Super::operator*(b); }
	
	Child operator/(const Child &b) const {
		const GenericVector &a = *this;
		Child c;
		for (int i = 0; i < size; ++i) {
			c.v[i] = a.v[i] / b.v[i];
		}
		return c;
	}
	
	Child operator/(const ScalarType &b) const { return Super::operator/(b); }
};

}
