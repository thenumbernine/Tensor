#include "TensorMath/Tensor.h"
#include "defs.h"
#include <algorithm>

int main() {

	typedef double Real;

#define COMMA ,	
	TEST_EQ(Tensor<Real COMMA Upper<3>>::numNestings, 1);
	TEST_EQ(Tensor<Real COMMA Upper<3>>::WriteIndexInfo<0>::size, 3);	
	
	TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::numNestings, 2);
	TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::WriteIndexInfo<0>::size, 5);
	TEST_EQ(Tensor<Real COMMA Upper<5> COMMA Upper<6>>::WriteIndexInfo<1>::size, 6);
	
	TEST_EQ(Tensor<Real COMMA Symmetric<Upper<3> COMMA Upper<3>>>::numNestings, 1);
	TEST_EQ(Tensor<Real COMMA Symmetric<Upper<3> COMMA Upper<3>>>::WriteIndexInfo<0>::size, 6);	//3*(3+1)/2

	{
		Tensor<Real, Upper<3>> a(1);
		Tensor<Real, Lower<3>> b(2);

		TEST_EQ(a, Tensor<Real COMMA Upper<3>>(1)); 
		TEST_EQ(b, Tensor<Real COMMA Lower<3>>(2));	

		Index i;
		a(i) = b(i);

		TEST_EQ(a, Tensor<Real COMMA Upper<3>>(2));

		static_assert(std::is_same<Tensor<Real, Upper<3>, Upper<3>>::DerefType, Vector<int,2>>::value, "values aren't same");
		Index j;
		Tensor<Real, Upper<3>, Upper<3>> m(
		[&](Vector<int,2> i) -> Real {
			return Real(i(0) == i(1));
		});
		//m(i,j) = m(j,i); 
	}
}

