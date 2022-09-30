#include "Tensor/Tensor.h"
#include "Tensor/Inverse.h"
#include "Common/Test.h"

// static tests here:

void test_vec() {

	//vector

	{
		// default ctor
		Tensor::float3 f;
		for (int i = 0; i < f.dim<0>; ++i) {
			TEST_EQ(f.s[i], 0);
		}
	}

	{
		// parenthesis ctor
		Tensor::float3 f(4,5,7);
	
		//operator<< works?
		std::cout << f << std::endl;
		// to_string works?
		std::cout << std::to_string(f) << std::endl;

		//test .x .y .z
		TEST_EQ(f.x, 4);
		TEST_EQ(f.y, 5);
		TEST_EQ(f.z, 7);
		// test .s0 .s1 .s2
		TEST_EQ(f.s0, 4);
		TEST_EQ(f.s1, 5);
		TEST_EQ(f.s2, 7);
		// test .s[]
		TEST_EQ(f.s[0], 4);
		TEST_EQ(f.s[1], 5);
		TEST_EQ(f.s[2], 7);
		// test () indexing
		TEST_EQ(f(0), 4);
		TEST_EQ(f(1), 5);
		TEST_EQ(f(2), 7);
		
		//test [] indexing
		TEST_EQ(f[0], 4);
		TEST_EQ(f[1], 5);
		TEST_EQ(f[2], 7);
		
		//.dims
		TEST_EQ(f.dims(), 3);
		TEST_EQ(f.dim<0>, 3);
	
		//iterator
		{
			auto i = f.begin();
			TEST_EQ(*i, 4); ++i;
			TEST_EQ(*i, 5); i++;
			TEST_EQ(*i, 7); i++;
			TEST_EQ(i, f.end());
		
			for (auto & i : f) {
				std::cout << "f iter = " << i << std::endl;
			}
			for (auto const & i : f) {
				std::cout << "f iter = " << i << std::endl;
			}
			// TODO verify cbegin/cend
			// TODO support for rbegin/rend const/not const and crbegin/crend
		}
		
		//lambda ctor
		TEST_EQ(f, Tensor::float3([](int i) -> float { return 4 + i * (i + 1) / 2; }));

		// TODO casting ctor

		// scalar ctor
		TEST_EQ(Tensor::float3(3), Tensor::float3(3,3,3));

		// bracket ctor
		Tensor::float3 g = {7,1,2};

		// vector/scalar operations
		TEST_EQ(f+1.f, Tensor::float3(5,6,8));
		TEST_EQ(f-1.f, Tensor::float3(3,4,6));
		TEST_EQ(f*12.f, Tensor::float3(48, 60, 84));
		TEST_EQ(f/2.f, Tensor::float3(2.f, 2.5f, 3.5f));
		// scalar/vector operations
		TEST_EQ(1.f+f, Tensor::float3(5,6,8));
		TEST_EQ(1.f-f, Tensor::float3(-3, -4, -6));
		TEST_EQ(12.f*f, Tensor::float3(48, 60, 84));
		TEST_EQ(2.f/f, Tensor::float3(0.5, 0.4, 0.28571428571429));
		// vector/vector operations
		TEST_EQ(f+g, Tensor::float3(11, 6, 9));
		TEST_EQ(f-g, Tensor::float3(-3, 4, 5));
		TEST_EQ(f*g, Tensor::float3(28, 5, 14));	//the controversial default-per-element-mul
		TEST_EQ(f/g, Tensor::float3(0.57142857142857, 5.0, 3.5));	// wow, this equality passes while sqrt(90) fails
		// hmm, do I have vector*vector => scalar default to dot product?

		// op= scalar
		{ Tensor::float3 h = f; h += 2; TEST_EQ(h, Tensor::float3(6,7,9)); }
		{ Tensor::float3 h = f; h -= 3; TEST_EQ(h, Tensor::float3(1,2,4)); }
		{ Tensor::float3 h = f; h *= 3; TEST_EQ(h, Tensor::float3(12,15,21)); }
		{ Tensor::float3 h = f; h /= 4; TEST_EQ(h, Tensor::float3(1,1.25,1.75)); }
		// op= vector
		{ Tensor::float3 h = f; h += Tensor::float3(3,2,1); TEST_EQ(h, Tensor::float3(7,7,8)); }
		{ Tensor::float3 h = f; h -= Tensor::float3(5,0,9); TEST_EQ(h, Tensor::float3(-1,5,-2)); }
		{ Tensor::float3 h = f; h *= Tensor::float3(-1,1,-2); TEST_EQ(h, Tensor::float3(-4,5,-14)); }
		{ Tensor::float3 h = f; h /= Tensor::float3(-2,3,-4); TEST_EQ(h, Tensor::float3(-2, 5.f/3.f, -1.75)); }

		// dot product
		TEST_EQ(dot(f,g), 47)
		TEST_EQ(f.dot(g), 47)
		
		// length-squared
		TEST_EQ(f.lenSq(), 90);
		TEST_EQ(lenSq(f), 90);

		// length
		TEST_EQ_EPS(f.length(), sqrt(90), 1e-6);
		TEST_EQ_EPS(length(f), sqrt(90), 1e-6);
		
		// cros product
		TEST_EQ(cross(f,g), Tensor::float3(3, 41, -31))

#if 0
		// outer product
		// hmm, in the old days macros couldn't detect <>'s so you'd have to wrap them in ()'s if the <>'s had ,'s in them
		// now same persists for {}'s it seems
		TEST_EQ(outerProduct(f,g), Tensor::float3x3(
			{28, 4, 8},
			{35, 5, 10},
			{49, 7, 14}
		));
#endif

		// TODO vector subset access
	
		// swizzle
		// TODO need an operator== between T and reference_wrapper<T> ...
		// or casting ctor?
		// a generic ctor between _vecs would be nice, but maybe problematic for _mat = _sym
		TEST_EQ(Tensor::float3(f.zyx()), Tensor::float3(7,5,4));
		TEST_EQ(Tensor::float2(f.xy()), Tensor::float2(4,5));
		TEST_EQ(Tensor::float2(f.yx()), Tensor::float2(5,4));
		TEST_EQ(Tensor::float2(f.yy()), Tensor::float2(5,5));
		
		/* more tests ...
		float2 float4
		int2 int3 int4
		
		default-template vectors of dif sizes (5 maybe? ... )
			assert that no .x exists to verify
		*/

		// verify vec4 list constructor works
		Tensor::float4 h = {2,3,4,5};
		TEST_EQ(h.x, 2);
		TEST_EQ(h.y, 3);
		TEST_EQ(h.z, 4);
		TEST_EQ(h.w, 5);

		Tensor::_vec<float,5> j = {5,6,7,8,9};
		//non-specialized: can't use xyzw for dim>4
		TEST_EQ(j[0], 5);
		TEST_EQ(j[1], 6);
		TEST_EQ(j[2], 7);
		TEST_EQ(j[3], 8);
		TEST_EQ(j[4], 9);
	
		// iterator copy
		Tensor::float3 f2;
		std::copy(f2.begin(), f2.end(), f.begin()); // crashing ...
		TEST_EQ(f, f2);
	
		// iterator copy from somewhere else
		std::array<float, 3> fa;
		std::copy(fa.begin(), fa.end(), f.begin());
		TEST_EQ(fa[0], f[0]);
		TEST_EQ(fa[1], f[1]);
		TEST_EQ(fa[2], f[2]);
	}

	// matrix
	{
		//bracket ctor
		Tensor::float3x3 m = {
			{1,2,3},
			{4,5,6},
			{7,8,9},
		};
		
		static_assert(m.rank == 2);
		static_assert(m.dim<0> == 3);
		static_assert(m.dim<1> == 3);

		// TODO .x .y .z indexing
		// TODO .s0 .s1 .s2 indexing
		// TODO .s[] indexing
		
		//m[i][j] indexing
		TEST_EQ(m[0][0], 1);
		TEST_EQ(m[0][1], 2);
		TEST_EQ(m[0][2], 3);
		TEST_EQ(m[1][0], 4);
		TEST_EQ(m[1][1], 5);
		TEST_EQ(m[1][2], 6);
		TEST_EQ(m[2][0], 7);
		TEST_EQ(m[2][1], 8);
		TEST_EQ(m[2][2], 9);

		//m(i)(j) indexing
		TEST_EQ(m(0)(0), 1);
		TEST_EQ(m(0)(1), 2);
		TEST_EQ(m(0)(2), 3);
		TEST_EQ(m(1)(0), 4);
		TEST_EQ(m(1)(1), 5);
		TEST_EQ(m(1)(2), 6);
		TEST_EQ(m(2)(0), 7);
		TEST_EQ(m(2)(1), 8);
		TEST_EQ(m(2)(2), 9);
		
		//m(i,j) indexing
		TEST_EQ(m(0,0), 1);
		TEST_EQ(m(0,1), 2);
		TEST_EQ(m(0,2), 3);
		TEST_EQ(m(1,0), 4);
		TEST_EQ(m(1,1), 5);
		TEST_EQ(m(1,2), 6);
		TEST_EQ(m(2,0), 7);
		TEST_EQ(m(2,1), 8);
		TEST_EQ(m(2,2), 9);

		//m(int2(i,j)) indexing
		TEST_EQ(m(Tensor::int2(0,0)), 1);
		TEST_EQ(m(Tensor::int2(0,1)), 2);
		TEST_EQ(m(Tensor::int2(0,2)), 3);
		TEST_EQ(m(Tensor::int2(1,0)), 4);
		TEST_EQ(m(Tensor::int2(1,1)), 5);
		TEST_EQ(m(Tensor::int2(1,2)), 6);
		TEST_EQ(m(Tensor::int2(2,0)), 7);
		TEST_EQ(m(Tensor::int2(2,1)), 8);
		TEST_EQ(m(Tensor::int2(2,2)), 9);
		
		// sub-vectors (row)
		TEST_EQ(m[0], Tensor::float3(1,2,3));
		TEST_EQ(m[1], Tensor::float3(4,5,6));
		TEST_EQ(m[2], Tensor::float3(7,8,9));

		//dims and rank.  really these are static_assert's, except dims(), but it could be, but I'd have to constexpr some things ...
		TEST_EQ(m.rank, 2);
		TEST_EQ(m.dims(), Tensor::int2(3,3));
		TEST_EQ(m.dim<0>, 3);
		TEST_EQ(m.dim<1>, 3);

		// read iterator
		{
			auto i = m.begin();
#if 0 // not sure which to use
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
#endif
#if 1
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
#endif
		}

		// write iterator (should match read iterator except for symmetric members)

		// lambda constructor
		// row-major, sequential in memory:
		TEST_EQ(m, Tensor::float3x3([](Tensor::int2 i) -> float { return 1 + i(1) + 3 * i(0); }));
		// col-major, sequential in memory:
		//TEST_EQ(m, Tensor::float3x3([](Tensor::int2 i) -> float { return 1 + i(0) + 3 * i(1); }));
		
		// TODO casting ctor

		// scalar ctor
		TEST_EQ(Tensor::float3x3(3), (Tensor::float3x3{{3,3,3},{3,3,3},{3,3,3}}));

		// TODO matrix subset access

		// TODO matrix swizzle

		//determinant
		TEST_EQ(determinant(m), 0);
		// TODO opertors
		// TODO make sure operator* works
		// TODO I don't think I have marix *= working yet	
		// TODO verify matrix swizzle

		TEST_EQ(m + 0.f, m);
		TEST_EQ(m * 1.f, m);
		TEST_EQ(m * 0.f, decltype(m)());

		auto m2 = matrixCompMult(m,m);
		for (int i = 0; i < m.dim<0>; ++i) {
			for (int j = 0; j < m.dim<1>; ++j) {
				TEST_EQ(m2(i,j), m(i,j) * m(i,j));
			}
		}
	}

	// tensor of vec-vec-vec
	{
		using T = Tensor::_tensor<float, 2, 4, 5>;
		T t;
		static_assert(t.rank == 3);
		static_assert(t.dim<0> == 2);
		static_assert(t.dim<1> == 4);
		static_assert(t.dim<2> == 5);

		//TODO 
		//tensor * scalar
		//scalar * tensor
	
		TEST_EQ(t + 0.f, t);
		TEST_EQ(t * 1.f, t);
		TEST_EQ(t * 0.f, decltype(t)());
	}

	//symmetric
	{
		auto a = Tensor::float3s3(); // default
		static_assert(a.rank == 2);
		static_assert(a.dim<0> == 3);
		static_assert(a.dim<1> == 3);
		// default ctor
		for (int i = 0; i < a.dim<0>; ++i) {
			for (int j = 0; j < a.dim<1>; ++j) {
				TEST_EQ(a(i,j), 0);
				TEST_EQ(a(j,i), 0);
			}
		}

		//TODO verify lambda ctor only covers write iterators, i.e. 6 inits instead of 9
		// lambda ctor using int,int...
		auto b = Tensor::float3s3([](int i, int j) -> float {
			return (float)(i+j);
		});
		for (int i = 0; i < b.dim<0>; ++i) {
			for (int j = 0; j < b.dim<1>; ++j) {
				TEST_EQ(b(i,j), i+j);
				TEST_EQ(b(j,i), i+j);
			}
		}

		{
			int k = 0;
			auto c = Tensor::float3s3([&](int i, int j) -> float {
				++k;
				return (float)(i+j);
			});
			TEST_EQ(k, 6);	//for write iterators in lambda ctor ...
			TEST_EQ(c, b);
		}

		// lambda ctor using int2
		TEST_EQ(b, Tensor::float3s3([](Tensor::int2 ij) -> float {
			return (float)(ij.x+ij.y);
		}));

		// test symmetry
		b(0,2) = 7;
		TEST_EQ(b(2,0), 7);
		b.xy = -1;
		TEST_EQ(b.yx, -1);

		// partial index
		for (int i = 0; i < b.dim<0>; ++i) {
			for (int j = 0; j < b.dim<1>; ++j) {
				TEST_EQ(b[i][j], b(i,j));
			}
		}
		
		TEST_EQ(a + 0.f, a);
		TEST_EQ(a * 1.f, a);
		TEST_EQ(a * 0.f, decltype(a)());
	
		// TODO how do GLSL matrix ctor from scalars work? 
		// do they initialize to full scalars like vecs do?
		// do they initialize to ident times scalar like math do?
	}
	
	// tensor with intermixed non-vec types:
	// tensor of vec-symmetric
	{
		//this is a T_ijk = T_ikj, i spans 3 dims, j and k span 2 dims
		using T2S3 = Tensor::_tensori<float, Tensor::index_vec<2>, Tensor::index_sym<3>>;
		
		// list ctor
		T2S3 t = {
			{1,2,3,4,5,6}, //xx xy yy xz yz zz
			{7,8,9,0,1,2},
		};
		// xyz field access
		TEST_EQ(t.x.xx, 1);
		TEST_EQ(t.x.xy, 2);
		TEST_EQ(t.x.yy, 3);
		TEST_EQ(t.x.xz, 4);
		TEST_EQ(t.x.yz, 5);
		TEST_EQ(t.x.zz, 6);
		TEST_EQ(t.y.xx, 7);
		TEST_EQ(t.y.xy, 8);
		TEST_EQ(t.y.yy, 9);
		TEST_EQ(t.y.xz, 0);
		TEST_EQ(t.y.yz, 1);
		TEST_EQ(t.y.zz, 2);
	
		// nested (int,int,int) access
		TEST_EQ(t(0,0,0), 1);
		TEST_EQ(t(0,0,1), 2);
		TEST_EQ(t(0,1,1), 3);
		TEST_EQ(t(0,0,2), 4);
		TEST_EQ(t(0,1,2), 5);
		TEST_EQ(t(0,2,2), 6);
		TEST_EQ(t(1,0,0), 7);
		TEST_EQ(t(1,0,1), 8);
		TEST_EQ(t(1,1,1), 9);
		TEST_EQ(t(1,0,2), 0);
		TEST_EQ(t(1,1,2), 1);
		TEST_EQ(t(1,2,2), 2);
	
		TEST_EQ(t + 0.f, t);
		TEST_EQ(t * 1.f, t);
		TEST_EQ(t * 0.f, decltype(t)());
	}
	// tensor of symmetric-vec
}
