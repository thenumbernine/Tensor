#include "Test/Test.h"

void test_TotallyAntisymmetric() {
	using float3 = Tensor::float3;
	using float3a3a3 = Tensor::_asymR<float, 3, 3>;
	static_assert(sizeof(float3a3a3) == sizeof(float));
	
	{
		auto L = float3a3a3();
		L(0,0,0) = 1;
		TEST_EQ(L(0,0,0), 0);
		L(0,0,1) = 1;
		TEST_EQ(L(0,0,1), 0);
		L(0,1,0) = 1;
		TEST_EQ(L(0,1,0), 0);
		L(1,0,0) = 1;
		TEST_EQ(L(1,0,0), 0);
		
		L(0,1,2) = 1;
		TEST_EQ(L(0,1,2), 1);
		TEST_EQ(L(1,2,0), 1);
		TEST_EQ(L(2,0,1), 1);
		TEST_EQ(L(2,1,0), -1);
		TEST_EQ(L(1,0,2), -1);
		TEST_EQ(L(0,2,1), -1);
	}

// parity flipping chokes the decltype(auto)
#if 0
	{
		auto f = [](int i, int j, int k) -> float {
			return sign(j-i) * sign(k-j);
		};
		auto t = float3a3a3(f);
		verifyAccessRank3<decltype(t)>(t, f);
		verifyAccessRank3<decltype(t) const>(t, f);
		operatorScalarTest(t);
	}
#endif

#if 0 // mul not working yet
	{
		auto L = float3a3a3(1);	//Levi-Civita permutation tensor
		auto x = float3(1,0,0);
		auto dualx = L * x;
		auto y = float3(0,1,0);
		auto z = dualx * y;
		ECHO(z);
	}
#endif
}
