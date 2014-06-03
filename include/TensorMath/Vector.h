#pragma once

#include <ostream>
#include "TensorMath/GenericVector.h"

/*
vector class for fixed-size templated dimension (i.e. size) and type
*/
template<typename Type_, int dim_>
struct Vector : public GenericVector<Type_, dim_, Type_, Vector<Type_, dim_> > {
	typedef GenericVector<Type_, dim_, Type_, Vector<Type_, dim_> > Parent;
	
	typedef typename Parent::Type Type;
	enum { dim = Parent::size };	//size is the size of the static vector, which coincides with the dim in the vector class (not so in matrix classes)
	typedef typename Parent::ScalarType ScalarType;

	//inherited constructors
	using Parent::Parent;

	template<typename Type2>
	operator Vector<Type2,dim>() const {
		Vector<Type2,dim> result;
		for (int i = 0; i < dim; ++i) {
			result.v[i] = (Type2)Parent::v[i];
		}
		return result;
	}
};

template<typename Type, int dim>
std::ostream &operator<<(std::ostream &o, const Vector<Type,dim> &t) {
	const char *sep = "";
	o << "(";
	for (int i = 0; i < t.dim; ++i) {
		o << sep << t(i);
		sep = ", ";
	}
	o << ")";
	return o;
}

