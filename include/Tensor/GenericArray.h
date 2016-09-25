#pragma once

#include "Tensor/clamp.h"
#include "Tensor/Meta.h"
#include "Common/crtp_cast.h"
#include <iostream>

namespace Tensor {

/*
the 'parent' curious pattern whatever for generic_vector and generic_matrix
	Type_ is the type of each element
	size_ is the size of the dense vector
	scalar_type_ is the type of the innermost nested scalars (in the event that elements are GenericArray's themselves)
	child is the curious reoccurring chlid class that uses this
*/
template<typename Type_, int size_, typename ScalarType_, typename Child>
struct GenericArray {
	typedef Type_ Type;
	enum { size = size_ };
	typedef ScalarType_ ScalarType;

	Type v[size];

	//default ctor: init all to zero
	GenericArray() {
		ForLoop<0, size, UnaryOp<Assign<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstScalarAccess<Type>>::template Inner>::exec(v, Type());
	}

	//default copy ctor
	GenericArray(const Child &a) {
		ForLoop<0, size, UnaryOp<Assign<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(v, a.v);
	}

	//default fill ctor
	GenericArray(const Type &x) {
		ForLoop<0, size, UnaryOp<Assign<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstScalarAccess<Type>>::template Inner>::exec(v, x);
	}

	//subset
	//I would like to return the child type, but this is by definition returning a different static size
	// and child types are flexible in their size initialization (rank-1's vs rank-2's, symmetric vs antisymmetric, etc.)
	//so I have to return the parent class type
	template<int start, int length>
	GenericArray<Type, length, ScalarType, Child> subset() const {
		static_assert(length <= size, "cannot get a subset that is greater than the array itself");
		static_assert(start >= 0, "cannot get negative bounds subsets");
		static_assert(start <= size - length, "cannot get subsets that span past the end");
		GenericArray<Type, length, ScalarType, Child> s;
		ForLoop<0, length, UnaryOp<Assign<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(s.v, v + start);
		return s;
	}	

	//bounds
	static Child clamp(const Child &a, const Child &min, const Child &max) {
		Child b;
		ForLoop<0, size, TernaryOp<Assign<Type, Type>, Clamp<Type>, ArrayAccess<Type>, ConstArrayAccess<Type>, ConstArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(b.v, a.v, min.v, max.v);
		return b;
	}

	//boolean operations

	template<typename ChildType2>
	bool operator==(const GenericArray<Type, size, ScalarType, ChildType2> &b) const {
		bool result = true;
		ForLoop<0, size, BinaryOp<AndInto<bool, bool>, Equals<bool, Type, Type>, ScalarAccess<bool>, ConstArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(result, v, b.v);
		return result;
	}

	bool operator!=(const Child &b) const { return !this->operator==(b); }

	//unary math operator

	Child operator-() const {
		Child b;
		ForLoop<0, size, UnaryOp<Assign<Type, Type>, UnaryMinus<Type>, ArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(b.v, v);
		return b;
	}

	//pairwise math operators

	Child operator+(const Child &b) const {
		Child c;
		ForLoop<0, size, BinaryOp<Assign<Type, Type>, Add<Type, Type, Type>, ArrayAccess<Type>, ConstArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(c.v, v, b.v);
		return c;
	}
	
	Child operator-(const Child &b) const {
		Child c;
		ForLoop<0, size, BinaryOp<Assign<Type, Type>, Sub<Type, Type, Type>, ArrayAccess<Type>, ConstArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(c.v, v, b.v);
		return c;
	}

	//I'm omitting * and / since matrix defines them separately
	//what if I overloaded them?

	//scalar math operators

	//I'm omitting scalar + and -, but not for any good reason

	Child operator*(const ScalarType &b) const {
		Child c;
		ForLoop<0, size, BinaryOp<Assign<Type, Type>, Mul<Type, Type, ScalarType>, ArrayAccess<Type>, ConstArrayAccess<Type>, ConstScalarAccess<ScalarType>>::template Inner>::exec(c.v, v, b);
		return c;
	}

	Child operator/(const ScalarType &b) const {
		Child c;
		ForLoop<0, size, BinaryOp<Assign<Type, Type>, Div<Type, Type, ScalarType>, ArrayAccess<Type>, ConstArrayAccess<Type>, ConstScalarAccess<ScalarType>>::template Inner>::exec(c.v, v, b);
		return c;
	}

	//these will have to be overridden to return the correct type (unless I cast it here...)

	Child &operator+=(const Child &b) {
		ForLoop<0, size, UnaryOp<AddInto<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(v, b.v);
		return crtp_cast<Child>(*this);
	}

	Child &operator-=(const Child &b) {
		ForLoop<0, size, UnaryOp<SubInto<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstArrayAccess<Type>>::template Inner>::exec(v, b.v);
		return crtp_cast<Child>(*this);
	}

	Child &operator*=(const ScalarType &b) {
		ForLoop<0, size, UnaryOp<MulInto<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstScalarAccess<Type>>::template Inner>::exec(v, b);
		return crtp_cast<Child>(*this);
	}

	Child &operator/=(const ScalarType &b) {
		ForLoop<0, size, UnaryOp<DivInto<Type, Type>, Identity<Type>, ArrayAccess<Type>, ConstScalarAccess<Type>>::template Inner>::exec(v, b);
		return crtp_cast<Child>(*this);
	}
};

};

