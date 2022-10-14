#include "Test/Test.h"

void test_Identity() {
	using floatI3 = Tensor::_ident<float, 3>;
	
	{
		auto I = floatI3(1);
		// ident == ident works
		TEST_EQ(I, floatI3(1));
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
		TEST_NE(I, floatI3(2));
		// ident != matrix works
			// off-diagonal
		TEST_EQ(I, (Tensor::float3x3{
			{1,2,3},
			{0,1,0},
			{0,0,1},
		}));
			// on-diagonal
		TEST_EQ(I, (Tensor::float3x3{
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
		auto I = floatI3(1);
		auto Iplus1 = I + 1;
		static_assert(std::is_same_v<decltype(Iplus1), Tensor::float3s3>);
		TEST_EQ(Iplus1, (Tensor::float3x3{
			{2,1,1},
			{1,2,1},
			{1,1,2},
		}));
	}
	{
		auto I = floatI3(1);
		auto Iplus1 = 1 + I;
		static_assert(std::is_same_v<decltype(Iplus1), Tensor::float3s3>);
		TEST_EQ(Iplus1, (Tensor::float3x3{
			{2,1,1},
			{1,2,1},
			{1,1,2},
		}));
	}

#if 0 //TODO
	// sym + ident => ident
	{
		auto I = floatI3(1);
		auto S = Tensor::float3s3(2);
		auto R = I + S;
		static_assert(std::is_same_v<decltype(R), Tensor::float3s3>);
	}
	{
		auto I = floatI3(1);
		auto S = Tensor::float3s3(2);
		auto R = S + I;
		static_assert(std::is_same_v<decltype(R), Tensor::float3s3>);
	}
#endif
}
