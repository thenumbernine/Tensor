#include "Test/Test.h"

void test_TotallySymmetric() {
	using float3s3s3 = Tensor::_symR<float, 3, 3>;
	auto s = float3s3s3();

	s(0,0,0) = 1;
	TEST_EQ(s(0,0,0), 1);

	s(1,0,0) = 2;
	TEST_EQ(s(1,0,0), 2);
	TEST_EQ(s(0,1,0), 2);
	TEST_EQ(s(0,0,1), 2);
}
