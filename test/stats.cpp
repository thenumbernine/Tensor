#include "Tensor/Tensor.h"
#include "Common/Test.h"
#include <typeinfo>
#include <iostream>

using namespace Tensor;

#define COMMA ,
int main() {
	typedef double Real;

	Tensor<Real,Upper<3>> a;
	ECHO(typeid(Tensor<Real COMMA Upper<3>>).name());
	TEST_EQ(Tensor<Real COMMA Upper<3>>::rank, 1);
	TEST_EQ(Tensor<Real COMMA Upper<3>>::TensorStats::rank, 1);
	TEST_EQ(Tensor<Real COMMA Upper<3>>::TensorStats::Index::rank, 1);
	TEST_EQ(Tensor<Real COMMA Upper<3>>::TensorStats::Index::dim, 3);
	TEST_EQ(Tensor<Real COMMA Upper<3>>::IndexInfo<0>::dim, 3);
	
	//this can be generated at compile time
	//I need a mechanism for retrieving per-element access to 'size' at compile time
	ECHO(a.size());

	Tensor<Real,Lower<4>> b;
	TEST_EQ(Tensor<Real COMMA Lower<4>>::rank, 1);
	TEST_EQ(Tensor<Real COMMA Lower<4>>::IndexInfo<0>::dim, 4);
	ECHO(b.size());

	Tensor<Real,Symmetric<Lower<3>,Lower<3>>> c;
	TEST_EQ(Tensor<Real COMMA Symmetric<Lower<3> COMMA Lower<3>>>::rank, 2);
	TEST_EQ(Tensor<Real COMMA Symmetric<Lower<3> COMMA Lower<3>>>::IndexInfo<0>::dim, 3);
	TEST_EQ(Tensor<Real COMMA Symmetric<Lower<3> COMMA Lower<3>>>::IndexInfo<1>::dim, 3);
	ECHO(c.size());

	Tensor<Real,Upper<5>,Upper<6>> d;
	TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::rank, 2);
	TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::IndexInfo<0>::dim, 5);
	TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::IndexInfo<1>::dim, 6);
	ECHO(d.size());

	Tensor<Real, Upper<4>, Symmetric<Upper<3>, Upper<3>>> e;
	TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::rank, 3);
	TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::IndexInfo<0>::dim, 4);
	TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::IndexInfo<1>::dim, 3);
	TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::IndexInfo<2>::dim, 3);
	ECHO(e.size());

	Tensor<Real, Antisymmetric<Lower<2>, Lower<2>>, Antisymmetric<Lower<3>, Lower<3>>> f;
	TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::rank, 4);
	TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::IndexInfo<0>::dim, 2);
	TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::IndexInfo<1>::dim, 2);
	TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::IndexInfo<2>::dim, 3);
	TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::IndexInfo<3>::dim, 3);
	ECHO(f.size());
	
}

