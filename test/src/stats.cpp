#include "Tensor/v1/Tensor.h"
#include "Common/Test.h"
#include <typeinfo>
#include <iostream>

namespace Tensor {
using namespace Tensor::v1;
}

void test_stats() {
	using Real = double;

	Tensor::Tensor<Real,Tensor::Upper<3>> a;
	ECHO(typeid(Tensor::Tensor<Real, Tensor::Upper<3>>).name());
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<3>>::rank), 1);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<3>>::TensorStats::rank), 1);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<3>>::TensorStats::Index::rank), 1);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<3>>::TensorStats::Index::dim), 3);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<3>>::IndexInfo<0>::dim), 3);
	
	//this can be generated at compile time
	//I need a mechanism for retrieving per-element access to 'size' at compile time
	ECHO(a.size());

	Tensor::Tensor<Real,Tensor::Lower<4>> b;
	TEST_EQ((Tensor::Tensor<Real, Tensor::Lower<4>>::rank), 1);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Lower<4>>::IndexInfo<0>::dim), 4);
	ECHO(b.size());

	Tensor::Tensor<Real,Tensor::Symmetric<Tensor::Lower<3>,Tensor::Lower<3>>> c;
	TEST_EQ((Tensor::Tensor<Real, Tensor::Symmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::rank), 2);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Symmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::IndexInfo<0>::dim), 3);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Symmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::IndexInfo<1>::dim), 3);
	ECHO(c.size());

	Tensor::Tensor<Real,Tensor::Upper<5>,Tensor::Upper<6>> d;
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<5>, Tensor::Upper<6>>::rank), 2);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<5>, Tensor::Upper<6>>::IndexInfo<0>::dim), 5);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<5>, Tensor::Upper<6>>::IndexInfo<1>::dim), 6);
	ECHO(d.size());

	Tensor::Tensor<Real, Tensor::Upper<4>, Tensor::Symmetric<Tensor::Upper<3>, Tensor::Upper<3>>> e;
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<4>, Tensor::Symmetric<Tensor::Upper<3>, Tensor::Upper<3>>>::rank), 3);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<4>, Tensor::Symmetric<Tensor::Upper<3>, Tensor::Upper<3>>>::IndexInfo<0>::dim), 4);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<4>, Tensor::Symmetric<Tensor::Upper<3>, Tensor::Upper<3>>>::IndexInfo<1>::dim), 3);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Upper<4>, Tensor::Symmetric<Tensor::Upper<3>, Tensor::Upper<3>>>::IndexInfo<2>::dim), 3);
	ECHO(e.size());

	Tensor::Tensor<Real, Tensor::Antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>, Tensor::Antisymmetric<Tensor::Lower<3>, Tensor::Lower<3>>> f;
	TEST_EQ((Tensor::Tensor<Real, Tensor::Antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>, Tensor::Antisymmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::rank), 4);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>, Tensor::Antisymmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::IndexInfo<0>::dim), 2);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>, Tensor::Antisymmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::IndexInfo<1>::dim), 2);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>, Tensor::Antisymmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::IndexInfo<2>::dim), 3);
	TEST_EQ((Tensor::Tensor<Real, Tensor::Antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>, Tensor::Antisymmetric<Tensor::Lower<3>, Tensor::Lower<3>>>::IndexInfo<3>::dim), 3);
	ECHO(f.size());
}
