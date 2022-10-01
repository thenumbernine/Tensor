void test_v1_basics();
void test_v1_iter();
void test_v1_stats();
void test_NegativeRef();
void test_Quat();
void test_Tensor();

int main() {
	test_v1_basics();
	test_v1_iter();
	test_v1_stats();
	test_NegativeRef();
	test_Quat();
	test_Tensor();
}
