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
	typedef tensor<real, Upper<3>> vector3;
*/
template<int dim>
struct Upper : public GenericRank1<dim> {};

/*
dim = dimension of this lower index
used like so:
	typedef tensor<real, Lower<3>> oneform3;
*/
template<int dim>
struct Lower : public GenericRank1<dim> {};

/*
used like so: 
	typedef tensor<real, Symmetric<Lower<3>, Lower<3>>> metric3;
	typedef tensor<real, Symmetric<Lower<3>, Lower<3>>, Symmetric<Lower<3>, Lower<3>>> riemann_metric3;
*/
template<typename Index1, typename Index2>
struct Symmetric {
	static_assert(Index1::dim == Index2::dim, "Symmetric can only accept two indexes of equal dimension");
	enum { dim = Index1::dim };
	enum { rank = Index1::rank + Index2::rank };
	
	template<typename InnerType, typename ScalarType>
	struct Body : public GenericSymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>> {
		typedef GenericSymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>> Parent;
		Body() : Parent() {}
		Body(const Body &b) : Parent(b) {}
		Body(const InnerType &t) : Parent(t) {}
	};
};

template<typename Index1, typename Index2>
struct Antisymmetric {
	static_assert(Index1::dim == Index2::dim, "Symmetric can only accept two indexes of equal dimension");
	enum { dim = Index1::dim };
	enum { rank = Index1::rank + Index2::rank };

	template<typename InnerType, typename ScalarType>
	struct Body : public GenericAntisymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>> {
		typedef GenericAntisymmetricMatrix<InnerType, Index1::dim, ScalarType, Body<InnerType, ScalarType>> Parent;
		Body() : Parent() {}
		Body(const Body &b) : Parent(b) {}
		Body(const InnerType &t) : Parent(t) {}
	};
};

};

