#include "Test/Test.h"

void test_Identity() {
	using float3i3 = Tensor::float3i3;
	
	{
		auto I = float3i3(1);
		// ident == ident works
		TEST_EQ(I, float3i3(1));
		// ident == matrix works
		TEST_EQ(I, (Tensor::float3x3{
			{1,0,0},
			{0,1,0},
			{0,0,1},
		}));
		// ident == sym works
		TEST_EQ(I, (Tensor::float3s3)(Tensor::float3x3{
			{1,0,0},
			{0,1,0},
			{0,0,1},
		}));

		// ident != ident works
		TEST_NE(I, float3i3(2));
		// ident != matrix works
			// off-diagonal
		TEST_NE(I, (Tensor::float3x3{
			{1,2,3},
			{0,1,0},
			{0,0,1},
		}));
			// on-diagonal
		TEST_NE(I, (Tensor::float3x3{
			{1,0,0},
			{0,2,0},
			{0,0,3},
		}));
		// ident != sym works
			// off-diagonal
		TEST_NE(I, (Tensor::float3s3)(Tensor::float3x3{
			{1,2,3},
			{0,1,0},
			{0,0,1},
		}));
			// on-diagonal
		TEST_NE(I, (Tensor::float3s3)(Tensor::float3x3{
			{1,0,0},
			{0,2,0},
			{0,0,3},
		}));
		// ident != asym works
		TEST_NE(I, Tensor::float3a3());
		TEST_NE(I, Tensor::float3a3(1,2,3));
	}

	// ident + scalar => sym
	{
		auto I = float3i3(1);
		auto Iplus1 = I + 1;
		static_assert(std::is_same_v<decltype(Iplus1), Tensor::float3s3>);
		TEST_EQ(Iplus1, (Tensor::float3x3{
			{2,1,1},
			{1,2,1},
			{1,1,2},
		}));
	}
	{
		auto I = float3i3(1);
		auto Iplus1 = 1 + I;
		static_assert(std::is_same_v<decltype(Iplus1), Tensor::float3s3>);
		TEST_EQ(Iplus1, (Tensor::float3x3{
			{2,1,1},
			{1,2,1},
			{1,1,2},
		}));
	}
	
	// TODO should this be ident? or maybe TensorRank4?  or maybe a separate test for correct operator results?
	using float3z3 = Tensor::_tensori<float, Tensor::index_zero<3>, Tensor::index_zero<3>>;

	// zero + zero = zero
	{
		auto Z = float3z3();
		auto Z2 = float3z3();
		auto R = Z + Z2;
		static_assert(std::is_same_v<decltype(R), float3z3>);
		TEST_EQ(R, (Tensor::float3x3{{0,0,0},{0,0,0},{0,0,0}}));
	}

	// ident + zero = ident
	{
		auto I = float3i3(1);
		auto Z = float3z3();
		auto R = I + Z;
		static_assert(std::is_same_v<decltype(R), float3i3>);
		TEST_EQ(R, (Tensor::float3x3{{1,0,0},{0,1,0},{0,0,1}}));
	}

	// ident + ident = ident
	{
		auto I1 = float3i3(1);
		auto I2 = float3i3(2);
		auto R = I1 + I2;
		static_assert(std::is_same_v<decltype(R), float3i3>);
		TEST_EQ(R, (Tensor::float3x3{{3,0,0},{0,3,0},{0,0,3}}));
	}

	// sym + ident => ident
	{
		auto I = float3i3(1);
		auto S = Tensor::float3s3(2);
		auto R = I + S;
		static_assert(std::is_same_v<decltype(R), Tensor::float3s3>);
	}
	{
		auto I = float3i3(1);
		auto S = Tensor::float3s3(2);
		auto R = S + I;
		static_assert(std::is_same_v<decltype(R), Tensor::float3s3>);
	}
	
	// ident-ident + ident-ident = ident-ident
	{
		using namespace Tensor;
		auto a = _tensori<float, index_ident<3>, index_ident<3>>();
		auto b = _tensori<float, index_ident<3>, index_ident<3>>();
		auto c = a + b;
		static_assert(std::is_same_v<decltype(c), _tensori<float, index_ident<3>, index_ident<3>>>);
	}
	
	// ident-ident + ident-sym = ident-sym
	{
		using namespace Tensor;
		auto a = _tensori<float, index_ident<3>, index_ident<3>>();
		auto b = _tensori<float, index_ident<3>, index_sym<3>>();
		auto c = a + b;
		static_assert(std::is_same_v<decltype(c), _tensori<float, index_ident<3>, index_sym<3>>>);
	}
	
	// ident-ident + ident-asym = ident-mat
	{
		using namespace Tensor;
		auto a = _tensori<float, index_ident<3>, index_ident<3>>();
		auto b = _tensori<float, index_ident<3>, index_asym<3>>();
		auto c = a + b;
		static_assert(std::is_same_v<decltype(c), _tensori<float, index_ident<3>, index_vec<3>, index_vec<3>>>);
	}
}
