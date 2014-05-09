#include "Common/Exception.h"
#include "TensorMath/Tensor.h"

#include <typeinfo>
#include <iostream>

using namespace std;

#define ECHO(x)	cout << #x << "\t" << x << endl;
#define TEST_EQ(a,b) cout << #a << " == " << #b << " :: " << a << " == " << b << endl; if (a != b) throw Exception() << "failed";

#define COMMA ,
int main() {
	try {
		typedef double Real;

		Tensor<Real,Upper<3>> a;
		ECHO(typeid(Tensor<Real COMMA Upper<3>>).name());
		TEST_EQ(Tensor<Real COMMA Upper<3>>::rank, 1);
		TEST_EQ(Tensor<Real COMMA Upper<3>>::TensorStats::rank, 1);
		TEST_EQ(Tensor<Real COMMA Upper<3>>::TensorStats::Index::rank, 1);
		TEST_EQ(Tensor<Real COMMA Upper<3>>::TensorStats::Index::dim, 3);
		TEST_EQ(Tensor<Real COMMA Upper<3>>::Index<0>::dim, 3);
		
		//this can be generated at compile time
		//I need a mechanism for retrieving per-element access to 'size' at compile time
		ECHO(a.size());

		Tensor<Real,Lower<4>> b;
		TEST_EQ(Tensor<Real COMMA Lower<4>>::rank, 1);
		TEST_EQ(Tensor<Real COMMA Lower<4>>::Index<0>::dim, 4);
		ECHO(b.size());

		Tensor<Real,Symmetric<Lower<3>,Lower<3>>> c;
		TEST_EQ(Tensor<Real COMMA Symmetric<Lower<3> COMMA Lower<3>>>::rank, 2);
		TEST_EQ(Tensor<Real COMMA Symmetric<Lower<3> COMMA Lower<3>>>::Index<0>::dim, 3);
		TEST_EQ(Tensor<Real COMMA Symmetric<Lower<3> COMMA Lower<3>>>::Index<1>::dim, 3);
		ECHO(c.size());

		Tensor<Real,Upper<5>,Upper<6>> d;
		TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::rank, 2);
		TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::Index<0>::dim, 5);
		TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::Index<1>::dim, 6);
		ECHO(d.size());

		Tensor<Real, Upper<4>, Symmetric<Upper<3>, Upper<3>>> e;
		TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::rank, 3);
		TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::Index<0>::dim, 4);
		TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::Index<1>::dim, 3);
		TEST_EQ(Tensor<Real COMMA Upper<4> COMMA Symmetric<Upper<3> COMMA Upper<3>>>::Index<2>::dim, 3);
		ECHO(e.size());

		Tensor<Real, Antisymmetric<Lower<2>, Lower<2>>, Antisymmetric<Lower<3>, Lower<3>>> f;
		TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::rank, 4);
		TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::Index<0>::dim, 2);
		TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::Index<1>::dim, 2);
		TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::Index<2>::dim, 3);
		TEST_EQ(Tensor<Real COMMA Antisymmetric<Lower<2> COMMA Lower<2>> COMMA Antisymmetric<Lower<3> COMMA Lower<3>>>::Index<3>::dim, 3);
		ECHO(f.size());
	} catch (std::exception &e) {
		cerr << e.what();
		return 1;
	}
	return 0;
}

