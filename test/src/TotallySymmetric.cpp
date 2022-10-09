#include "Test/Test.h"

void test_TotallySymmetric() {
	/*
	unique indexing of 3s3s3:
000
001 010 100
002 020 200
011 101 110
012 021 102 120 201 210
022 202 220
111
112 121 211
122 212 221
222
	*/
	using float3s3s3 = Tensor::_symR<float, 3, 3>;
	static_assert(sizeof(float3s3s3) == sizeof(float) * 10);
	{
		auto t = float3s3s3();

		t(0,0,0) = 1;
		TEST_EQ(t(0,0,0), 1);

		t(1,0,0) = 2;
		TEST_EQ(t(1,0,0), 2);
		TEST_EQ(t(0,1,0), 2);
		TEST_EQ(t(0,0,1), 2);

		// Iterator needs operator()(intN i)
		auto i = t.begin();
		TEST_EQ(*i, 1);
	}

	{
		auto f = [](int i, int j, int k) -> float { return i + j + k; };
		auto t = float3s3s3(f);
		verifyAccessRank3<decltype(t)>(t, f);
		verifyAccessRank3<decltype(t) const>(t, f);

		// operators need Iterator
		operatorScalarTest(t);
	}

	{
		auto f = [](int i, int j, int k) -> float { return i + j + k; };
		auto t = float3s3s3(f);

		auto v = Tensor::float3{3, 6, 9};
		auto m = v * t;
		// TODO ExpandIthIndex of a start or end index should preserve symmetry of the rest
		static_assert(std::is_same_v<decltype(m), Tensor::float3x3>);
	}
}
