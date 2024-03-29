#include "Tensor/AntiSymRef.h"
#include "Common/Test.h"

static_assert(Common::is_instance_v<Tensor::AntiSymRef<double>, Tensor::AntiSymRef>);
static_assert(Common::is_instance_v<Tensor::AntiSymRef<Tensor::AntiSymRef<double>>, Tensor::AntiSymRef>);

void test_AntiSymRef() {
	float f;

	f = 123;
	auto n = Tensor::AntiSymRef<float>(f, Tensor::Sign::NEGATIVE);
	TEST_EQ(f, 123);
	TEST_EQ(n, -123);
	n = 456;
	TEST_EQ(n, 456);
	TEST_EQ(f, -456);
	
	f = 123;
	auto p = Tensor::AntiSymRef<float>(f, Tensor::Sign::POSITIVE);
	TEST_EQ(f, 123);
	TEST_EQ(p, 123);
	p = 456;
	TEST_EQ(p, 456);
	TEST_EQ(f, 456);
	
	//auto p = Tensor::AntiSymRef<float>(f, Tensor::AntiSymRef::POSITIVE);
	auto z = Tensor::AntiSymRef<float>();
	TEST_EQ(z, 0);
	z = 1;
	TEST_EQ(z, 0);
}
