#pragma once

#include "Tensor/clamp.h"
#include "Common/crtp_cast.h"

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
		for (int i = 0; i < size; ++i) {
			v[i] = Type();
		}
	}

	//default copy ctor
	GenericArray(const Child &a) {
		for (int i = 0; i < size; ++i) {
			v[i] = a.v[i];
		}
	}

	//default fill ctor
	GenericArray(const Type &x) {
		for (int i = 0; i < size; ++i) {
			v[i] = x;
		}
	}

	//subset
	template<int start, int length>
	GenericArray<Type, length, ScalarType, Child> subset() {	//bastard child.  this will make vec2s who think their CRAP type is vec3
		static_assert(length <= size, "cannot get a subset that is greater than the array itself");
		static_assert(start >= 0, "cannot get negative bounds subsets");
		static_assert(start <= size - length, "cannot get subsets that span past the end");
		GenericArray<Type, length, ScalarType, Child> s;
		for (int i = 0; i < length; ++i) {
			s.v[i] = v[start+i];
		}
		return s;
	}	

	//bounds
	static Child clamp(const Child &a, const Child &min, const Child &max) {
		Child b;
		for (int i = 0; i < size; ++i) {
			b.v[i] = clamp(a.v[i], min.v[i], max.v[i]);
		}
		return b;
	}

	//boolean operations

	template<typename ChildType2>
	bool operator==(const GenericArray<Type, size, ScalarType, ChildType2> &b) const {
		const GenericArray &a = *this;
		for (int i = 0; i < size; ++i) {
			if (a.v[i] != b.v[i]) return false;
		}
		return true;
	}

	bool operator!=(const Child &b) const { return ! this->operator==(b); }

	//unary math operator

	Child operator-() const {
		const GenericArray &a = *this;
		Child b;
		for (int i = 0; i < size; ++i) {
			b.v[i] = -a.v[i];
		}
		return b;
	}

	//pairwise math operators

	Child operator+(const Child &b) const {
		const GenericArray &a = *this;
		Child c;
		for (int i = 0; i < size; ++i) {
			c.v[i] = a.v[i] + b.v[i];
		}
		return c;
	}
	
	Child operator-(const Child &b) const {
		const GenericArray &a = *this;
		Child c;
		for (int i = 0; i < size; ++i) {
			c.v[i] = a.v[i] - b.v[i];
		}
		return c;
	}

	//I'm omitting * and / since matrix defines them separately
	//what if I overloaded them?

	//scalar math operators

	//I'm omitting scalar + and -, but not for any good reason

	Child operator*(const ScalarType &b) const {
		const GenericArray &a = *this;
		Child c;
		for (int i = 0; i < size; ++i) {
			c.v[i] = a.v[i] * b;
		}
		return c;
	}


	Child operator/(const ScalarType &b) const {
		const GenericArray &a = *this;
		Child c;
		for (int i = 0; i < size; ++i) {
			c.v[i] = a.v[i] / b;
		}
		return c;
	}

	//these will have to be overridden to return the correct type (unless I cast it here...)
	
	Child &operator+=(const Child &b) {
		for (int i = 0; i < size; ++i) {
			v[i] += b.v[i];
		}
		return crtp_cast<Child>(*this);
	}

	Child &operator-=(const Child &b) {
		for (int i = 0; i < size; ++i) {
			v[i] -= b.v[i];
		}
		return crtp_cast<Child>(*this);
	}

	Child &operator*=(const ScalarType &b) {
		for (int i = 0; i < size; ++i) {
			v[i] *= b;
		}
		return crtp_cast<Child>(*this);
	}

	Child &operator/=(const ScalarType &b) {
		for (int i = 0; i < size; ++i) {
			v[i] /= b;
		}
		return crtp_cast<Child>(*this);
	}
};

};

