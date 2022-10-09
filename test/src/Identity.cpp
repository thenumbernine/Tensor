#include "Test/Test.h"

void test_Identity() {
	using floatI3 = Tensor::_ident<float, 3>;
	{
		auto I = floatI3(1);
		TEST_EQ(I, (Tensor::_mat<float,3,3>{
			{1,0,0},
			{0,1,0},
			{0,0,1},
		}));
	}

	{
		auto I = floatI3(1);
		auto Iplus1 = I + 1;
		static_assert(std::is_same_v<decltype(Iplus1), Tensor::float3s3>);
		TEST_EQ(Iplus1, (Tensor::_mat<float,3,3>{
			{2,1,1},
			{1,2,1},
			{1,1,2},
		}));
	}

}
