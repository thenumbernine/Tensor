#pragma once

#include "Tensor/GenericVector.h"
#include <ostream>

namespace Tensor {

/*
vector class for fixed-size templated dimension (i.e. size) and type
*/
template<typename Type_, int dim_>
struct Vector : public GenericVector<Type_, dim_, Type_, Vector<Type_, dim_> > {
	using Parent = GenericVector<Type_, dim_, Type_, Vector<Type_, dim_> >;
	
	using Type = typename Parent::Type;
	static constexpr auto dim = Parent::size;	//size is the size of the static vector, which coincides with the dim in the vector class (not so in matrix classes)
	using ScalarType = typename Parent::ScalarType;

	//inherited constructors
	using Parent::Parent;

	template<typename Type2, int dim2>
	operator Vector<Type2,dim2>() const {
		Vector<Type2,dim2> result;
		int i = 0;
		for (; i < (dim < dim2 ? dim : dim2); ++i) {
			result.v[i] = (Type2)Parent::v[i];
		}
		for (; i < dim2; ++i) {
			result.v[i] = Type();
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

using bool2 = Vector<bool, 2>;
using bool3 = Vector<bool, 3>;
using bool4 = Vector<bool, 4>;

using char2 = Vector<char, 2>;
using char3 = Vector<char, 3>;
using char4 = Vector<char, 4>;

using uchar2 = Vector<unsigned char, 2>;
using uchar3 = Vector<unsigned char, 3>;
using uchar4 = Vector<unsigned char, 4>;

using short2 = Vector<short, 2>;
using short3 = Vector<short, 3>;
using short4 = Vector<short, 4>;

using ushort2 = Vector<unsigned short, 2>;
using ushort3 = Vector<unsigned short, 3>;
using ushort4 = Vector<unsigned short, 4>;

using int2 = Vector<int, 2>;
using int3 = Vector<int, 3>;
using int4 = Vector<int, 4>;

using uint2 = Vector<unsigned int, 2>;
using uint3 = Vector<unsigned int, 3>;
using uint4 = Vector<unsigned int, 4>;

using float2 = Vector<float, 2>;
using float3 = Vector<float, 3>;
using float4 = Vector<float, 4>;

using double2 = Vector<double, 2>;
using double3 = Vector<double, 3>;
using double4 = Vector<double, 4>;

}
