#include "Tensor/Tensor.h"
#include "Tensor/Inverse.h"
#include "Common/Test.h"

// static tests here:

void test_vec() {
	//vector

	{
		// default ctor
		Tensor::float3 f;
		for (int i = 0; i < f.dim; ++i) {
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
		TEST_EQ(f.ith_dim<0>, 3);
	
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
	
		// outer product
		// hmm, in the old days macros couldn't detect <>'s so you'd have to wrap them in ()'s if the <>'s had ,'s in them
		// now same persists for {}'s it seems
		TEST_EQ(outerProduct(f,g), Tensor::float3x3(
			{28, 4, 8},
			{35, 5, 10},
			{49, 7, 14}
		));

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
		//can't use xyzw for dim>4
		TEST_EQ(j[0], 5);
		TEST_EQ(j[1], 6);
		TEST_EQ(j[2], 7);
		TEST_EQ(j[3], 8);
		TEST_EQ(j[4], 9);
	}

	// matrix
	{
		//bracket ctor
		Tensor::float3x3 m = {
			{1,2,3},
			{4,5,6},
			{7,8,9},
		};

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
		TEST_EQ(m.ith_dim<0>, 3);
		TEST_EQ(m.ith_dim<1>, 3);

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
	}

	/*
	matrix

	opertors
	make sure operator* works
	I don't think I have marix *= working yet
	
	symmetric

	tensor of vec-symmetric
	tensor of symmetric-vec
	*/

	{
		Tensor::float3x3 m;
	
		//TODO verify lambda ctor only covers write operators, i.e. 6 inits instead of 9
		
		static_assert(m.rank == 2);
		static_assert(m.ith_dim<0> == 3);
		static_assert(m.ith_dim<1> == 3);
	}

	// tensor of vec-vec-vec
	{
		using T = Tensor::_tensor<float, 2, 4, 5>;
		T t;
		static_assert(t.rank == 3);
		static_assert(t.ith_dim<0> == 2);
		static_assert(t.ith_dim<1> == 4);
		static_assert(t.ith_dim<2> == 5);
	
		//tensor * scalar
		//scalar * tensor
	
//		t * 1.f;
	}

}
