#include "Test/Test.h"

namespace Test {
	using namespace Common;
	using namespace Tensor;
	using namespace std;
	static_assert(tuple_size_v<float3x3::InnerPtrTensorTuple> == 2);
	static_assert(is_same_v<
		tuple_element_t<0, float3x3::InnerPtrTensorTuple>,
		float3x3*
	>);
	static_assert(is_same_v<
		tuple_element_t<1, float3x3::InnerPtrTensorTuple>,
		float3*
	>);
	static_assert(tuple_size_v<float3x3::InnerPtrTuple> == 3);
	static_assert(is_same_v<
		tuple_element_t<0, float3x3::InnerPtrTuple>,
		float3x3*
	>);
	static_assert(is_same_v<
		tuple_element_t<1, float3x3::InnerPtrTuple>,
		float3*
	>);
	static_assert(is_same_v<
		tuple_element_t<2, float3x3::InnerPtrTuple>,
		float*
	>);
	static_assert(float3x3::dimseq::size() == std::tuple_size_v<float3x3::InnerPtrTensorTuple>);
	static_assert(float3x3::dimseq::size() == float3x3::rank);
	STATIC_ASSERT_EQ((seq_get_v<0, float3x3::dimseq>), 3);
	STATIC_ASSERT_EQ((seq_get_v<1, float3x3::dimseq>), 3);
}

void test_Matrix() {
	// matrix

	{	// 1x1
		using float1x1 = Tensor::tensor<float,1,1>;
		//list ctor
		auto x = float1x1{{3}};
		TEST_EQ(x(0,0), 3);
		using float1 = Tensor::floatN<1>;
		auto x1 = float1(2);
		// lambdas work
		{
			auto x2 = float1x1([&](int, int) -> float { return x1(0); });
			TEST_EQ(x2(0,0), 2);
		}
		{
			auto x2 = float1x1([&](Tensor::int2) -> float { return x1(0); });
			TEST_EQ(x2(0,0), 2);
		}
		{
			auto x2 = float1x1{x1};
			TEST_EQ(x2(0,0), 2);
		}
#if 0
		{	// failing to compile - ambiguous constructor
			auto x2 = float1x1(x1);
			TEST_EQ(x2(0,0), 2);
		}
#endif
	}

	{	//1x2
		// in fact, all list ctors of dim=(1,1,1.., N), for N != 1 are failing to compile
		using namespace Tensor;
		using float1x2 = tensor<float,1,2>;

		auto x1 = float2(3,4);
		auto x = float1x2{x1};
		{
			ECHO(x);
			TEST_EQ(x, float1x2{x1});
		}
		{
			auto y = float1x2{{3,4}};
			TEST_EQ(x,y);
			TEST_EQ(x, (float1x2{{3,4}}));
		}
#if 0
		{
			auto y = float1x2({3,4});
			TEST_EQ(x,y);		// errors ... initializes y to {{3,3}}
		}
#endif
#if 0
		TEST_EQ(x,(float1x2({3,4})));		// errors ... initializes y to {{3,3}}
#endif
#if 0
		TEST_EQ(x,(float1x2{3,4}));		// should fail to compile?  it doesn't - and does scalar ctor
#endif
#if 0
		{
			auto y = float1x2(x1);	// won't compile
			ECHO(y);
		}
#endif
#if 0
		ECHO(float1x2(x1));	// won't compile - ambiguous conversion
#endif
#if 0
		ECHO(float1x2(float2(3,4)));	// won't compile - ambiguous conversion
#endif
#if 0
		auto y = float1x2(float2(3,4));	// won't compile - ambiguous conversion
#endif
	}

	{	//2x1
		using namespace Tensor;
		using float1 = floatN<1>;
		using float2x1 = tensor<float,2,1>;

		ECHO(float2x1(float1(1),float1(2)));

		ECHO(float2x1({1},{2}));

		ECHO((float2x1{{1},{2}}));

		auto a1 = float1(1);
		auto a2 = float1(2);
		ECHO(float2x1(a1,a2));
		ECHO((float2x1{a1,a2}));

		auto a = float2x1{{1},{2}};
		TEST_EQ(a(0,0), 1);
		TEST_EQ(a(1,0), 2);

		auto b = float2x1({1},{2});
		TEST_EQ(a,b);

		auto c = float2x1(float1(1), float1(2));
		TEST_EQ(a,c);
	}

	{	//2x2 ctors
		using namespace Tensor;

		ECHO(float2x2(float2(1,2),float2(3,4)));

		ECHO(float2x2({1,2},{3,4}));

		ECHO((float2x2{{1,2},{3,4}}));

		auto a1 = float2(1,2);
		auto a2 = float2(3,4);
		ECHO(float2x2(a1,a2));
		ECHO((float2x2{a1,a2}));

		auto a = float2x2{{1,2},{3,4}};
		ECHO(a);

		auto b = float2x2({1,2},{3,4});
		ECHO(b);

		auto c = float2x2(float2(1,2),float2(3,4));
		ECHO(c);
	}

	//bracket ctor
	Tensor::float3x3 m = {
		{1,2,3},
		{4,5,6},
		{7,8,9},
	};

	//dims and rank.  really these are static_assert's, except dims, but it could be, but I'd have to constexpr some things ...
	STATIC_ASSERT_EQ(m.rank, 2);
	STATIC_ASSERT_EQ((m.dim<0>), 3);
	STATIC_ASSERT_EQ((m.dim<1>), 3);
	TEST_EQ(m.dims(), Tensor::int2(3,3));
	STATIC_ASSERT_EQ(m.numNestings, 2);
	STATIC_ASSERT_EQ((m.count<0>), 3);
	STATIC_ASSERT_EQ((m.count<1>), 3);

	// .x .y .z indexing
	TEST_EQ(m.x.x, 1);
	TEST_EQ(m.x.y, 2);
	TEST_EQ(m.x.z, 3);
	TEST_EQ(m.y.x, 4);
	TEST_EQ(m.y.y, 5);
	TEST_EQ(m.y.z, 6);
	TEST_EQ(m.z.x, 7);
	TEST_EQ(m.z.y, 8);
	TEST_EQ(m.z.z, 9);

	// .s0 .s1 .s2 indexing
	TEST_EQ(m.s0.s0, 1);
	TEST_EQ(m.s0.s1, 2);
	TEST_EQ(m.s0.s2, 3);
	TEST_EQ(m.s1.s0, 4);
	TEST_EQ(m.s1.s1, 5);
	TEST_EQ(m.s1.s2, 6);
	TEST_EQ(m.s2.s0, 7);
	TEST_EQ(m.s2.s1, 8);
	TEST_EQ(m.s2.s2, 9);

	// indexing - various [] and (int...) and (intN)
	auto f = [](int i, int j) -> float { return 1 + j + 3 * i; };
	verifyAccessRank2<decltype(m)>(m, f);
	verifyAccessRank2<decltype(m) const>(m, f);

	//matrix-specific access , doesn't work for sym or asym
	auto verifyAccessMat = []<typename T, typename F>(T & t, F f) {
		for (int i = 0; i < T::template dim<0>; ++i) {
			for (int j = 0; j < T::template dim<1>; ++j) {
				typename T::Scalar e = f(i,j);
				TEST_EQ(t.s[i](j), e);
				TEST_EQ(t.s[i][j], e);
				TEST_EQ(t(i).s[j], e);
				TEST_EQ(t[i].s[j], e);
				TEST_EQ(t.s[i].s[j], e);
			}
		}
	};
	verifyAccessMat.template operator()<decltype(m)>(m, f);
	verifyAccessMat.template operator()<decltype(m) const>(m, f);

	// scalar ctor
	// TODO how do GLSL matrix ctor from scalars work?
	// do they initialize to full scalars like vecs do?
	// do they initialize to ident times scalar like math do?
	TEST_EQ(Tensor::float3x3(3), (Tensor::float3x3{{3,3,3},{3,3,3},{3,3,3}}));
	TEST_EQ(Tensor::float3x3(3), (Tensor::float3x3({3,3,3},{3,3,3},{3,3,3})));

	// lambda constructor
	// row-major, sequential in memory:
	TEST_EQ(m, Tensor::float3x3([](Tensor::int2 i) -> float { return 1 + i(1) + 3 * i(0); }));
	// col-major, sequential in memory:
	//  why I don't do col-major? because then it's transposed of C construction, and that means m[i][j] == m.s[j].s[i] so your indexes are transposed the storage indexes
	//TEST_EQ(m, Tensor::float3x3([](Tensor::int2 i) -> float { return 1 + i(0) + 3 * i(1); }));

	// TODO casting ctor

	// read iterator
	{
		auto i = m.begin();
		if constexpr (Tensor::int2::useReadIteratorOuter) {
			// iterating in memory order for row-major
			// also in left-right top-bottom order when read
			// but you have to increment the last index first and first index last
			// TODO really?
			// but lambda init is now m(i,j) == 1 + i(1) + 3 * i(0) ... transposed of typical memory indexing
			TEST_EQ(*i, 1); ++i;
			TEST_EQ(*i, 2); ++i;
			TEST_EQ(*i, 3); ++i;
			TEST_EQ(*i, 4); ++i;
			TEST_EQ(*i, 5); ++i;
			TEST_EQ(*i, 6); ++i;
			TEST_EQ(*i, 7); ++i;
			TEST_EQ(*i, 8); ++i;
			TEST_EQ(*i, 9); ++i;
			TEST_EQ(i, m.end());
		} else {
			//iterating transpose to memory order for row-major
			// - inc first index first, last index last
			// but lambda init is now m(i,j) == 1 + i(0) + 3 * i(1)  ... typical memory indexing
			// I could fulfill both at the same time by making my matrices column-major, like OpenGL does ... tempting ...
			TEST_EQ(*i, 1); ++i;
			TEST_EQ(*i, 4); ++i;
			TEST_EQ(*i, 7); ++i;
			TEST_EQ(*i, 2); ++i;
			TEST_EQ(*i, 5); ++i;
			TEST_EQ(*i, 8); ++i;
			TEST_EQ(*i, 3); ++i;
			TEST_EQ(*i, 6); ++i;
			TEST_EQ(*i, 9); ++i;
			TEST_EQ(i, m.end());
		}
	}

	// write iterator (should match read iterator except for symmetric members)

	// sub-vectors (row)
	TEST_EQ(m[0], Tensor::float3(1,2,3));
	TEST_EQ(m[1], Tensor::float3(4,5,6));
	TEST_EQ(m[2], Tensor::float3(7,8,9));

	// TODO matrix subset access

	// TODO matrix swizzle

	// operators
	operatorScalarTest(m);
}
