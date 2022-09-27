#include "Tensor/Vector.h"
#include "Tensor/Inverse.h"
#include "Common/Test.h"

using namespace Tensor;

// static tests here:

void test_vec() {
	//vector

	{
		// default ctor
		float3 f;
		for (int i = 0; i < f.dim; ++i) {
			TEST_EQ(f.s[i], 0);
		}
	}

	{
		// parenthesis ctor
		float3 f(4,5,7);
	
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
			TEST_EQ(*i, 4);
			++i;
			TEST_EQ(*i, 5);
			i++;
			TEST_EQ(*i, 7);
			i++;
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

		// bracket ctor
		float3 g = {7,1,2};

		// vector/scalar operations
		TEST_EQ(f+1.f, float3(5,6,8));
		TEST_EQ(f-1.f, float3(3,4,6));
		TEST_EQ(f*12.f, float3(48, 60, 84));
		TEST_EQ(f/2.f, float3(2.f, 2.5f, 3.5f));
		// scalar/vector operations
		TEST_EQ(1.f+f, float3(5,6,8));
		TEST_EQ(1.f-f, float3(-3, -4, -6));
		TEST_EQ(12.f*f, float3(48, 60, 84));
		TEST_EQ(2.f/f, float3(0.5, 0.4, 0.28571428571429));
		// vector/vector operations
		TEST_EQ(f+g, float3(11, 6, 9));
		TEST_EQ(f-g, float3(-3, 4, 5));
		TEST_EQ(f*g, float3(28, 5, 14));	//the controversial default-per-element-mul
		TEST_EQ(f/g, float3(0.57142857142857, 5.0, 3.5));	// wow, this equality passes while sqrt(90) fails
		// hmm, do I have vector*vector => scalar default to dot product?

		// op= scalar 
		{ float3 h = f; h += 2; TEST_EQ(h, float3(6,7,9)); }
		{ float3 h = f; h -= 3; TEST_EQ(h, float3(1,2,4)); }
		{ float3 h = f; h *= 3; TEST_EQ(h, float3(12,15,21)); }
		{ float3 h = f; h /= 4; TEST_EQ(h, float3(1,1.25,1.75)); }
		// op= vector 
		{ float3 h = f; h += float3(3,2,1); TEST_EQ(h, float3(7,7,8)); }
		{ float3 h = f; h -= float3(5,0,9); TEST_EQ(h, float3(-1,5,-2)); }
		{ float3 h = f; h *= float3(-1,1,-2); TEST_EQ(h, float3(-4,5,-14)); }
		{ float3 h = f; h /= float3(-2,3,-4); TEST_EQ(h, float3(-2, 5.f/3.f, -1.75)); }

		// dot product
		TEST_EQ(dot(f,g), 47)
		TEST_EQ(f.dot(g), 47)
		
		// length-squared
		TEST_EQ(f.lenSq(), 90);
		TEST_EQ(lenSq(f), 90);

		// length
		TEST_EQ_EPS(f.length(), sqrt(90), 1e-7);
		TEST_EQ_EPS(length(f), sqrt(90), 1e-7);
		
		// cros product
		TEST_EQ(cross(f,g), float3(3, 41, -31))
	
		// outer product
		// hmm, in the old days macros couldn't detect <>'s so you'd have to wrap them in ()'s if the <>'s had ,'s in them
		// now same persists for {}'s it seems
		TEST_EQ(outerProduct(f,g), float3x3(
			{28, 4, 8},
			{35, 5, 10},
			{49, 7, 14}
		));

		// TODO vector subset access
	
		// swizzle
		auto fxxx = f.zyx();
		// TODO need an operator== between T and reference_wrapper<T> ...
		// or casting ctor?
		// a generic ctor between _vecs would be nice, but maybe problematic for _mat = _sym
		TEST_EQ(float3(f.zyx()), float3(7,5,4));
		TEST_EQ(float2(f.xy()), float2(4,5));
		TEST_EQ(float2(f.yx()), float2(5,4));
		TEST_EQ(float2(f.yy()), float2(5,5));
		
		/* more tests ...
		float2 float4
		int2 int3 int4
		
		default-template vectors of dif sizes (5 maybe? ... )
			assert that no .x exists to verify	
		*/
	}

	// matrix
	{
		//bracket ctor
		float3x3 m = {
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
		TEST_EQ(m(int2(0,0)), 1);
		TEST_EQ(m(int2(0,1)), 2);
		TEST_EQ(m(int2(0,2)), 3);
		TEST_EQ(m(int2(1,0)), 4);
		TEST_EQ(m(int2(1,1)), 5);
		TEST_EQ(m(int2(1,2)), 6);
		TEST_EQ(m(int2(2,0)), 7);
		TEST_EQ(m(int2(2,1)), 8);
		TEST_EQ(m(int2(2,2)), 9);
		
		// sub-vectors (row)
		TEST_EQ(m[0], float3(1,2,3));
		TEST_EQ(m[1], float3(4,5,6));
		TEST_EQ(m[2], float3(7,8,9));

		TEST_EQ(m.dims(), int2(3,3));
		TEST_EQ(m.ith_dim<0>, 3);
		TEST_EQ(m.ith_dim<1>, 3);

		// read iterator
		{
			auto i = m.begin();
			// iterating in memory order for row-major
			// also in left-right top-bottom order when read
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
		}

		// write iterator (should match read iterator except for symmetric members)

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

	tensor of vec-vec-vec
	
	symmetric

	tensor of vec-symmetric
	tensor of symmetric-vec
	*/
}
