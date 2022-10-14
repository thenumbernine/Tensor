#include "Tensor/Tensor.h"
#include "Common/Test.h"
#include <algorithm>

void test_Index() {
	using real = double;
	//index assignment
	{
		auto a = Tensor::_tensor<real, 3>(1);
		auto b = Tensor::_tensor<real, 3>(2);

		TEST_EQ(a.rank, 1);
		TEST_EQ(b.rank, 1);

		TEST_EQ(a, (Tensor::_tensor<real, 3>(1)));
		TEST_EQ(b, (Tensor::_tensor<real, 3>(2)));

		Tensor::Index<'i'> i;
		a(i) = b(i);

		TEST_EQ(a, (Tensor::_tensor<real, 3>(2)));
	}
	
	{
		//make sure 2D swizzling works
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::_tensor<real, 3, 3> m;
		m(1,0) = 1;
		ECHO(m);
		m(i,j) = m(j,i);
		TEST_EQ(m(0, 1), 1);
		ECHO(m);
	}

	{
		//make sure 3D swizzling works
		//this verifies the mapping between indexes in tensor assignment (since the 2D case is always a cycle of at most period 2, i.e. its own inverse)
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::Index<'k'> k;
		Tensor::_tensor<real, 3, 3, 3> s;
		s(0,1,0) = 1;
		ECHO(s);
		s(i,j,k) = s(j,k,i);	//s(0,0,1) = s(0,1,0)
		TEST_EQ(s(0,0,1), 1);
		ECHO(s);
	}

#if 0 // TODO
	{
		//arithemetic operations
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::_tensor<real, 3> b, c;
		Tensor::_tensor<real, 3, 3> a;

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
		real dot = b(i) * c(i);

		//matrix multiplication
		c(i) = a(i,j) * b(j);
	
		//discrete differentiation?
		c(i) = (a(i+1) - a(i-1)) / (2 * dx)
	}
#endif
}

int main() {
	test_Index();
}
