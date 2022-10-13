#include "Test/Test.h"

void test_Matrix() {	
	// matrix
	
	//bracket ctor
	Tensor::float3x3 m = {
		{1,2,3},
		{4,5,6},
		{7,8,9},
	};
	
	//dims and rank.  really these are static_assert's, except dims, but it could be, but I'd have to constexpr some things ...
	static_assert(m.rank == 2);
	static_assert(m.dim<0> == 3);
	static_assert(m.dim<1> == 3);
	TEST_EQ(m.dims(), Tensor::int2(3,3));
	static_assert(m.numNestings == 2);
	static_assert(m.count<0> == 3);
	static_assert(m.count<1> == 3);

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
		if constexpr (std::is_same_v<Tensor::int2::ReadInc<0>, Tensor::int2::ReadIncOuter<0>>) {
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
		} else if constexpr (std::is_same_v<Tensor::int2::ReadInc<0>, Tensor::int2::ReadIncInner<0>>) {
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
