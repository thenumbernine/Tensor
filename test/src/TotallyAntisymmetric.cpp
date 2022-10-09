#include "Test/Test.h"

void test_TotallyAntisymmetric() {
	using float3 = Tensor::float3;
	using float3a3a3 = Tensor::_asymR<float, 3, 3>;
	static_assert(sizeof(float3a3a3) == sizeof(float));
	auto L = float3a3a3(1);	//Levi-Civita permutation tensor

#if 0 // mul not working yet
	auto x = float3(1,0,0);
	auto dualx = L * x;
	auto y = float3(0,1,0);
	auto z = dualx * y;
	ECHO(z);
#endif
}
