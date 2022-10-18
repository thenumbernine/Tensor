#include "Tensor/Tensor.h"
#include "Common/Test.h"
#include <algorithm>

namespace TupleTests {
	using namespace std;
	using namespace Common;
	using namespace Tensor;

	using I = Index<'i'>;
	using J = Index<'j'>;
	I i;
	J j;

	static_assert(is_same_v<
		GatherIndexesImpl<
			decltype(float3()(i))::IndexTuple,
			make_integer_sequence<int, 1>
		>::indexes,
		tuple<I>
	>);

	static_assert(is_same_v<
		GatherIndexesImpl<
			decltype(float3()(i))::IndexTuple,
			make_integer_sequence<int, 1>
		>::type,
		tuple<
			pair<
				I,
				integer_sequence<int, 0>
			>
		>
	>);

	static_assert(is_same_v<
		GatherIndexesImpl<
			decltype(float3x3()(i,j))::IndexTuple,
			make_integer_sequence<int, 2>
		>::indexes,
		tuple<I, J>
	>);
	
	static_assert(is_same_v<
		decltype(float3x3()(i,j))::GatheredIndexes,
		tuple<
			pair<
				I,
				integer_sequence<int, 0>
			>,
			pair<
				J,
				integer_sequence<int, 1>
			>
		>
	>);

}

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
		ECHO(m);
		TEST_EQ(m(0, 1), 1);
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
		// explicitly-specified storage
		auto c = (.5 * (a(i,j) - a(j,i))).assignR<Tensor::double3a3>(i,j);
		static_assert(std::is_same_v<Tensor::double3a3, decltype(c)>);
		TEST_EQ(c, makeAsym(a));
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		//assignR
#if 0 // expected compile fail
		{
			Tensor::float3x3 a;
			auto c = a(i);
		}
#endif
		{
			Tensor::float3x3 a;
			auto c = a(i,j).assignR<Tensor::float3x3>(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x3>);
		}
		{
			Tensor::float3x3 a;
			auto c = a(i,j).assignR<Tensor::float3x3>(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x3>);
		}	
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assignR<Tensor::float2x3>(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assignR<Tensor::float3x2>(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x2>);
		}
		//assign
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assign(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assign(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x2>);
		}	
		{
			Tensor::float2x3 a;
			Tensor::float3x2 b;
			auto c = (a(j,i) + b(i,j)).assign(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x2>);
		}
		// good because I do support matching-rank, non-matching-dim tensor ctor 
		{
			Tensor::float2x3 a;
			Tensor::float3x3 c = a(j,i);	
		}
#if 0	// compile fail because ranks do not match
		// TODO put compile-fail tests in their own cpp file and have a script assert the compiler fails
		{
			Tensor::float3x3x3 d = a(j,i);
		}
#endif	
	}

// TODO DO enforce dimension constraints between expression operations
// and then require Index to specify subrank, or just grab the subset<> of the tensor.
// TODO sub-tensor casting, not just sub-vector.  return tensor-of-refs. 	
#if 0 // TODO
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 a = {{1,2,3},{4,5,6},{7,8,9}};

		// symmetrize using index notation
		Tensor::double3x3 b;
		b(i,j) = .5 * (a(i,j) + a(j,i));
		TEST_EQ(b, makeSym(a));
		// implicit storage type, for now picks the worst case
		auto c = (.5 * (a(i,j) - a(j,i))).assign(i,j);
		static_assert(std::is_same_v<Tensor::double3x3, decltype(c)>);
		TEST_EQ(c, makeAsym(a));
	}
	
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

int main() {
	test_Index();
}
