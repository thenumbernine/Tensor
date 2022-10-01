#include "Tensor/NegativeRef.h"
#include "Common/Test.h"

void test_NegativeRef() {
	float f = 123;
	auto n = Tensor::NegativeRef<float>(f);
	TEST_EQ(f, 123);
	TEST_EQ(n, -123);
	n = 456;
	TEST_EQ(n, 456);
	TEST_EQ(f, -456);
}
