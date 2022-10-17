#include "Test/Test.h"

namespace TestTotallySymmetric {
	
	static_assert(std::is_same_v<Tensor::float3a3::RemoveIndex<0>, Tensor::float3>);
	static_assert(std::is_same_v<Tensor::float3a3a3::RemoveIndex<0>, Tensor::float3a3>);
	// TODO get this to work
	static_assert(std::is_same_v<Tensor::float3a3a3::RemoveIndex<1>, Tensor::float3x3>);
	static_assert(std::is_same_v<Tensor::float3a3a3::RemoveIndex<2>, Tensor::float3a3>);


	using namespace Tensor;
	
	STATIC_ASSERT_EQ((constexpr_isqrt(0)), 0);
	STATIC_ASSERT_EQ((constexpr_isqrt(1)), 1);
	STATIC_ASSERT_EQ((constexpr_isqrt(2)), 1);
	STATIC_ASSERT_EQ((constexpr_isqrt(3)), 1);
	STATIC_ASSERT_EQ((constexpr_isqrt(4)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(5)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(6)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(7)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(8)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(9)), 3);

	static_assert(constexpr_factorial(0) == 1);
	static_assert(constexpr_factorial(1) == 1);
	static_assert(constexpr_factorial(2) == 2);
	static_assert(constexpr_factorial(3) == 6);
	static_assert(constexpr_factorial(4) == 24);

	static_assert(nChooseR(0,0) == 1);
	static_assert(nChooseR(1,0) == 1);
	static_assert(nChooseR(1,1) == 1);
	static_assert(nChooseR(2,0) == 1);
	static_assert(nChooseR(2,1) == 2);
	static_assert(nChooseR(2,2) == 1);
	static_assert(nChooseR(3,0) == 1);
	static_assert(nChooseR(3,1) == 3);
	static_assert(nChooseR(3,2) == 3);
	static_assert(nChooseR(3,3) == 1);
	static_assert(nChooseR(4,0) == 1);
	static_assert(nChooseR(4,1) == 4);
	static_assert(nChooseR(4,2) == 6);
	static_assert(nChooseR(4,3) == 4);
	static_assert(nChooseR(4,4) == 1);
}



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
		static_assert(std::is_same_v<decltype(m), Tensor::float3s3>);
	}

	// make sure call-through works
	{
		using namespace Tensor;
		// t_ijklmn = t_(ij)(klm)n
		auto f = [](int i, int j, int k, int l, int m, int n) -> float {
			return (i+j) - (k+l+m) + n;
		};
		auto t = _tensori<float, storage_sym<3>, storage_symR<3,3>, storage_vec<3>>(f);
		ECHO(t);
		ECHO(t(0,0,0,0,0,0));
	}
}
