#include "Tensor/Tensor.h"
#include "Common/Test.h"
#include <algorithm>

void test_Index() {
	//index assignment
	{
		auto a = Tensor::double3(1);
		auto b = Tensor::double3(2);

		TEST_EQ(a.rank, 1);
		TEST_EQ(b.rank, 1);

		TEST_EQ(a, (Tensor::double3(1)));
		TEST_EQ(b, (Tensor::double3(2)));

		Tensor::Index<'i'> i;
		a(i) = b(i);

		TEST_EQ(a, (Tensor::double3(2)));
	}
	
	{
		//make sure 2D swizzling works
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 m;
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
		Tensor::_tensor<double, 3, 3, 3> s;
		s(0,1,0) = 1;
		ECHO(s);
		s(i,j,k) = s(j,k,i);	//s(0,0,1) = s(0,1,0)
		TEST_EQ(s(0,0,1), 1);
		ECHO(s);
	}

	{
		//arithemetic operations
		Tensor::Index<'i'> i;
		
		Tensor::double3 a = {1,2,3};
		Tensor::double3 b = {5,7,11};

		Tensor::double3 c;
		c(i) = a(i) + b(i);
		TEST_EQ(c, (Tensor::double3(6,9,14)));
	}

	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 a = {{1,2,3},{4,5,6},{7,8,9}};

		// transpose
		a(i,j) = a(j,i);
		TEST_EQ(a, (Tensor::double3x3{{1,4,7},{2,5,8},{3,6,9}}));

		// add to transpose and self-assign
		a(i,j) = a(i,j) + a(j,i);
		TEST_EQ(a, (Tensor::double3x3{{2,6,10},{6,10,14},{10,14,18}}));
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 a = {{1,2,3},{4,5,6},{7,8,9}};

		// symmetrize using index notation
		Tensor::double3x3 b;
		b(i,j) = .5 * (a(i,j) + a(j,i));
		TEST_EQ(b, makeSym(a));
		auto c = (.5 * (a(i,j) - a(j,i))).assignR<Tensor::double3a3>(i,j);
		static_assert(std::is_same_v<Tensor::double3a3, decltype(c)>);
		TEST_EQ(c, makeAsym(a));
	}
// TODO DO enforce dimension constraints between expression operations
// and then require Index to specify subrank, or just grab the subset<> of the tensor.
// TODO sub-tensor casting, not just sub-vector.  return tensor-of-refs. 	
#if 0 // TODO
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		
		Tensor::double3 a = {1,2,3};
		Tensor::double3 b = {5,7,11};

		Tensor::double3x3 c;
		c(i,j) = a(i) * b(j);
	
	}
	{
		Tensor::Index<'i'> i;
		
		Tensor::double3 a = {1,2,3};
		Tensor::double3 b = {5,7,11};

		double c = a(i) * b(i);
	}
	{
		Tensor::double3x3 a;

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

#if 0
int main() {
	test_Index();
}
#endif
