#include "Test/Test.h"

void test_Matrix() {	
	// rank-2

	auto verifyAccessRank2 = []<typename T, typename F>(T & t, F f){
		for (int i = 0; i < T::template dim<0>; ++i) {
			for (int j = 0; j < T::template dim<1>; ++j) {
				typename T::Scalar x = f(i,j);
				TEST_EQ(t(i)(j), x);
				TEST_EQ(t(i,j), x);
				TEST_EQ(t(Tensor::int2(i,j)), x);
				TEST_EQ(t[i](j), x);
				TEST_EQ(t(i)[j], x);
				TEST_EQ(t[i][j], x);
				if constexpr (!std::is_const_v<T>) {
					t(i)(j) = x;
					t(i,j) = x;
					t(Tensor::int2(i,j)) = x;
					t[i](j) = x;
					t(i)[j] = x;
					t[i][j] = x;
				}
			}
		}
	};

	// matrix
	{
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
		TEST_EQ(m.dims, Tensor::int2(3,3));
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
		verifyAccessRank2.template operator()<decltype(m)>(m, f);
		verifyAccessRank2.template operator()<decltype(m) const>(m, f);
	
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

	//symmetric
	
	{
		auto a = Tensor::float3s3(); // default
		static_assert(a.rank == 2);
		static_assert(a.dim<0> == 3);
		static_assert(a.dim<1> == 3);
		// default ctor
		for (int i = 0; i < a.count<0>; ++i) {
			TEST_EQ(a.s[i], 0);
		}
		for (int i = 0; i < a.dim<0>; ++i) {
			for (int j = 0; j < a.dim<1>; ++j) {
				TEST_EQ(a(i,j), 0);
				TEST_EQ(a(j,i), 0);
			}
		}
/*
use a symmetric procedural matrix with distinct values , esp for verifying .x_x fields
don't use a_ij = i+j, because a_02 == a_11
so here's a procedural symmetric matrix with all distinct symmetric components:
a_ij = i*i + j*j
	{{0, 1, 4},
	{1, 2, 5},
	{4, 5, 8}}
so a.s == {0,1,2,4,5,8};
*/
		a = Tensor::float3s3(0,1,2,4,5,8);
		
		// verify index access works

		auto verifyAccessSym = []<typename T>(T & a){
			// testing fields
			TEST_EQ(a.x_x, 0);
			TEST_EQ(a.x_y, 1);
			TEST_EQ(a.x_z, 4);
			TEST_EQ(a.y_x, 1);
			TEST_EQ(a.y_y, 2);
			TEST_EQ(a.y_z, 5);
			TEST_EQ(a.z_x, 4);
			TEST_EQ(a.z_y, 5);
			TEST_EQ(a.z_z, 8);
		};
		verifyAccessSym.template operator()<decltype(a)>(a);
		verifyAccessSym.template operator()<decltype(a) const>(a);
		
		auto f = [](int i, int j) -> float { return i*i + j*j; };
		verifyAccessRank2.template operator()<decltype(a)>(a, f);
		verifyAccessRank2.template operator()<decltype(a) const>(a, f);

		// lambda ctor using int,int...
		TEST_EQ(a, Tensor::float3s3([](int i, int j) -> float { return i*i + j*j; }));
		
		// lambda ctor using int2
		TEST_EQ(a, Tensor::float3s3([](Tensor::int2 ij) -> float { return ij(0)*ij(0) + ij(1)*ij(1); }));
		
		// verify only 6 writes take place during ctor
		{
			int k = 0;
			// verifies lambda-of-ref too
			auto c = Tensor::float3s3([&](int i, int j) -> float {
				++k;
				return (float)(i*i+j*j);
			});
			TEST_EQ(k, 6);	//for write iterators in lambda ctor ...
			TEST_EQ(c, a);
		}

		// lambda ctor using int2
		TEST_EQ(a, Tensor::float3s3([](Tensor::int2 ij) -> float {
			return (float)(ij.x*ij.x+ij.y*ij.y);
		}));


		/*
		test storing matrix
		for this test, construct from an asymmetric matrix
			0 1 2
			3 4 5
			6 7 8
		... in a symmetric tensor
		if storage / write iterate is lower-triangular then this will be
		   0 3 6
		   3 4 7
		   6 7 8
		if it is upper-triangular:
			0 1 2
			1 4 5
			2 5 8
		*/
		auto b = Tensor::float3s3([](int i, int j) -> float {
			return 3 * i + j;
		});
#if 1 // upper triangular
		// test storage order
		// this order is for sym/asym getLocalReadForWriteIndex incrementing iread(0) first
		// it also means for _asym that i<j <=> POSITIVE, j<i <=> NEGATIVE
		TEST_EQ(b.s[0], 0);	// xx
		TEST_EQ(b.s[1], 3); // xy
		TEST_EQ(b.s[2], 4); // yy
		TEST_EQ(b.s[3], 6); // xz
		TEST_EQ(b.s[4], 7); // yz
		TEST_EQ(b.s[5], 8); // zz
		// test arg ctor
		TEST_EQ(b, Tensor::float3s3(0,3,4,6,7,8));
#else // lower triangular
		// test storage order
		// this order is for sym/asym getLocalReadForWriteIndex incrementing iread(1) first
		// it also means for _asym that i<j <=> NEGATIVE, j<i <=> POSITIVE
		TEST_EQ(b.s[0], 0);	// xx
		TEST_EQ(b.s[1], 1); // xy
		TEST_EQ(b.s[2], 4); // yy
		TEST_EQ(b.s[3], 2); // xz
		TEST_EQ(b.s[4], 5); // yz
		TEST_EQ(b.s[5], 8); // zz
		// test arg ctor
		TEST_EQ(b, Tensor::float3s3(0,1,4,2,5,8));
#endif

		// test symmetry read/write
		b(0,2) = 7;
		TEST_EQ(b(2,0), 7);
		b.x_y = -1;
		TEST_EQ(b.y_x, -1);

		// partial index
		for (int i = 0; i < b.dim<0>; ++i) {
			for (int j = 0; j < b.dim<1>; ++j) {
				TEST_EQ(b[i][j], b(i,j));
			}
		}
		
		// operators
		operatorScalarTest(a);
		operatorMatrixTest<Tensor::float3s3>();
	
		TEST_EQ(Tensor::trace(Tensor::float3s3({1,2,3,4,5,6})), 10);
	}

	// antisymmetric matrix
	{
		/*
		[ 0  1  2]
		[-1  0  3]
		[-2 -3  0]
		*/
		auto t = Tensor::float3a3{
			/*x_y=*/1,
			/*x_z=*/2,
			/*y_z=*/3
		};
		// (int,int) access
		t(0,0) = 1; //cannot write to diagonals
		TEST_EQ(t(0,0),0);
		t(1,1) = 2;
		TEST_EQ(t(1,1),0);
		t(2,2) = 3;
		TEST_EQ(t(2,2),0);

		// TODO indexes are 1 off
		auto a = Tensor::float3a3([](int i, int j) -> float { return i + j; });
		ECHO(a);
		TEST_EQ(a, t);

		auto b = Tensor::float3a3([](Tensor::int2 ij) -> float { return ij.x + ij.y; });
		ECHO(b);
		TEST_EQ(b, t);

		auto f = [](int i, int j) -> float { return sign(j-i)*(i+j); };
		
		auto verifyAccessAntisym = []<typename T>(T & t){
			// "field" method access
			TEST_EQ(t.x_x(), 0);
			TEST_EQ(t.x_y(), 1);
			TEST_EQ(t.x_z(), 2);
			TEST_EQ(t.y_x(), -1);
			TEST_EQ(t.y_y(), 0);
			TEST_EQ(t.y_z(), 3);
			TEST_EQ(t.z_x(), -2);
			TEST_EQ(t.z_y(), -3);
			TEST_EQ(t.z_z(), 0);
		};
		verifyAccessAntisym.template operator()<decltype(t)>(t);
		verifyAccessAntisym.template operator()<decltype(t) const>(t);
		
		verifyAccessRank2.template operator()<decltype(t)>(t, f);
		verifyAccessRank2.template operator()<decltype(t) const>(t, f);

		// verify antisymmetric writes work
		for (int i = 0; i < t.dim<0>; ++i) {
			for (int j = 0; j < t.dim<1>; ++j) {
				float k = 1 + i + j;
				t(i,j) = k;
				if (i != j) {
					TEST_EQ(t(i,j), k);
					TEST_EQ(t(j,i), -k);
				} else {
					TEST_EQ(t(i,j), 0);
				}
			}
		}

		// TODO verify that 'float3a3::ExpandStorage<0> == float3x3' & same with <1>

		// verify assignment to expanded type
		// TODO won't work until you get intN dereference in _asym
		Tensor::float3x3 c = t;
		TEST_EQ(c, (Tensor::float3x3{
			{0, -2, -3},
			{2, 0, -4},
			{3, 4, 0}
		}));
	
		//can't do yet until I fix asym access
		//operatorScalarTest(t);
		operatorMatrixTest<Tensor::float3a3>();
	}

	{
		Tensor::float3x3 m = Tensor::float3x3{
			{0,1,2},
			{-1,0,3},
			{-2,-3,0}
		};
		ECHO(m);
		Tensor::float3a3 as = m;
		ECHO(as);
		Tensor::float3x3 mas = as;
		ECHO(mas);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				TEST_EQ(as(i,j), m(i,j));
			}
		}
		TEST_EQ(as,m);
	}
}
