#pragma once

/*
holds different indexes for tensors
these are what go in the args... of the tensor class
*/

#include "Tensor/Vector.h"
#include "Tensor/GenericRank1.h"					//Upper, Lower
#include "Tensor/GenericSymmetricMatrix.h"			//Symmetric
#include "Tensor/GenericAntisymmetricMatrix.h"		//Antisymmetric

namespace Tensor {

/*
dim = dimension of this upper index
used like so: 
	using vector3 = Tensor<real, Upper<3>>;
*/
template<int dim>
struct Upper : public GenericRank1<dim> {};

/*
dim = dimension of this lower index
used like so:
	using oneform3 = Tensor<real, Lower<3>>;
*/
template<int dim>
struct Lower : public GenericRank1<dim> {};

/*
used like so: 
	using metric3 = Tensor<real, Symmetric<Lower<3>, Lower<3>>>;
	using riemann_metric3 = Tensor<real, Symmetric<Lower<3>, Lower<3>>, Symmetric<Lower<3>, Lower<3>>>;
*/
template<typename Index1, typename Index2>
struct Symmetric {
	static_assert(Index1::dim == Index2::dim, "Symmetric can only accept two indexes of equal dimension");
	static constexpr auto dim = Index1::dim;
	static constexpr auto rank = Index1::rank + Index2::rank;
	
	template<typename InnerType, typename ScalarType>
	struct Body : public GenericSymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>> {
		using Parent = GenericSymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>>;
		Body() : Parent() {}
		Body(const Body &b) : Parent(b) {}
		Body(const InnerType &t) : Parent(t) {}
	};
};

template<typename Index1, typename Index2>
struct Antisymmetric {
	static_assert(Index1::dim == Index2::dim, "Symmetric can only accept two indexes of equal dimension");
	static constexpr auto dim = Index1::dim;
	static constexpr auto rank = Index1::rank + Index2::rank;

	template<typename InnerType, typename ScalarType>
	struct Body : public GenericAntisymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>> {
		using Parent = GenericAntisymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>>;
		Body() : Parent() {}
		Body(const Body &b) : Parent(b) {}
		Body(const InnerType &t) : Parent(t) {}
	};
};

}
