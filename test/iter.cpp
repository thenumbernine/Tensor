#include "Tensor/Tensor.h"
#include "Common/Test.h"
#include <algorithm>

using namespace Tensor;

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

	//write iterator test
	{
		Tensor<Real, Symmetric<Upper<2>, Upper<2>>> s([&](Vector<int,2> i){
			return Real(i(0) + 2 * i(1));
		});
		TEST_EQ(s(0,0), 0);
		TEST_EQ(s(0,1), 1);
		TEST_EQ(s(1,1), 3);
		
		//would be '0' if write iter skipped a mem address present in the matrix
		//would be '2' if write iter was traversing a non-symmetric matrix
		TEST_EQ(s(1,0), 1);	
	}

	//index assignment
	{
		Tensor<Real, Upper<3>> a(1);
		Tensor<Real, Lower<3>> b(2);

		TEST_EQ(a, Tensor<Real COMMA Upper<3>>(1)); 
		TEST_EQ(b, Tensor<Real COMMA Lower<3>>(2));	

		Index<'i'> i;
		a(i) = b(i);

		TEST_EQ(a, Tensor<Real COMMA Upper<3>>(2));
	}
	
	{
		//make sure 2D swizzling works
		Index<'i'> i;
		Index<'j'> j;
		Tensor<Real, Upper<3>, Upper<3>> m;
		m(1,0) = 1;
		ECHO(m);
		m(i,j) = m(j,i);
		TEST_EQ(m(0, 1), 1);	
		ECHO(m);
	}

	{
		//make sure 3D swizzling works
		//this verifies the mapping between indexes in tensor assignment (since the 2D case is always a cycle of at most period 2, i.e. its own inverse)
		Index<'i'> i;
		Index<'j'> j;
		Index<'k'> k;
		Tensor<Real, Upper<3>, Upper<3>, Upper<3>> s;
		s(0,1,0) = 1;
		ECHO(s);
		s(i,j,k) = s(j,k,i);	//s(0,0,1) = s(0,1,0)
		TEST_EQ(s(0,0,1), 1);
		ECHO(s);
	}

	{
#if 0
		//arithemetic operations
		Index<'i'> i;
		Index<'j'> j;
		Tensor<Real, Upper<3>> b, c;
		Tensor<Real, Upper<3>, Lower<3>> a;

		b(0) = 1;
		b(1) = 2;
		b(2) = 3;
		c(0) = 5;
		c(1) = 7;
		c(2) = 11;

		//outer product
		a(i,j) = b(i) * c(j);

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				TEST_EQ(a(i,j), b(i) * c(j));
			}
		}

		//exterior product
		a(i,j) = b(i) * c(j) - b(j) * c(i);

		//inner product?
		Real dot = b(i) * c(i);

		//matrix multiplication
		c(i) = a(i,j) * b(j);
	
		//discrete differentiation?
		c(i) = (a(i+1) - a(i-1)) / (2 * dx)
#endif
	}
}
