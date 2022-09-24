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
std::ostream &operator<<(std::ostream &o, Vector<Type,dim> const &t) {
	char const *sep = "";
	o << "(";
	for (int i = 0; i < t.dim; ++i) {
		o << sep << t(i);
		sep = ", ";
	}
	o << ")";
	return o;
}

template<int dim> using boolN = Vector<bool, dim>;
using bool1 = boolN<1>;
using bool2 = boolN<2>;
using bool3 = boolN<3>;
using bool4 = boolN<4>;

template<int dim> using charN = Vector<char, dim>;
using char1 = charN<1>;
using char2 = charN<2>;
using char3 = charN<3>;
using char4 = charN<4>;

template<int dim> using ucharN = Vector<unsigned char, dim>;
using uchar1 = ucharN<1>;
using uchar2 = ucharN<2>;
using uchar3 = ucharN<3>;
using uchar4 = ucharN<4>;

template<int dim> using shortN = Vector<short, dim>;
using short1 = shortN<1>;
using short2 = shortN<2>;
using short3 = shortN<3>;
using short4 = shortN<4>;

template<int dim> using ushortN = Vector<unsigned short, dim>;
using ushort1 = ushortN<1>;
using ushort2 = ushortN<2>;
using ushort3 = ushortN<3>;
using ushort4 = ushortN<4>;

template<int dim> using intN = Vector<int, dim>;
using int1 = intN<1>;
using int2 = intN<2>;
using int3 = intN<3>;
using int4 = intN<4>;

template<int dim> using uintN = Vector<unsigned int, dim>;
using uint1 = uintN<1>;
using uint2 = uintN<2>;
using uint3 = uintN<3>;
using uint4 = uintN<4>;

template<int dim> using floatN = Vector<float, dim>;
using float1 = floatN<1>;
using float2 = floatN<2>;
using float3 = floatN<3>;
using float4 = floatN<4>;

template<int dim> using doubleN = Vector<double, dim>;
using double1 = doubleN<1>;
using double2 = doubleN<2>;
using double3 = doubleN<3>;
using double4 = doubleN<4>;

template<typename T> using _vec1 = Vector<T, 1>;
template<typename T> using _vec2 = Vector<T, 2>;
template<typename T> using _vec3 = Vector<T, 3>;
template<typename T> using _vec4 = Vector<T, 4>;

}
