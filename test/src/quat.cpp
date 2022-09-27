#include "Tensor/Quat.h"
#include "Common/Test.h"

using namespace Tensor;

void test_quat() {
	quatf q;
	TEST_EQ(q.x, 0);
	TEST_EQ(q.y, 0);
	TEST_EQ(q.z, 0);
	TEST_EQ(q.w, 1);
}
