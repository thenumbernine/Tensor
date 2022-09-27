#include "Tensor/Vector.h"
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
		
		// bracket ctor
		float3 g = {7,1,2};
		//test [] indexing
		TEST_EQ(g[0], 7);
		TEST_EQ(g[1], 1);
		TEST_EQ(g[2], 2);

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
	
		// swizzle
		auto fxxx = f.zyx();
		// TODO need an operator== between T and reference_wrapper<T> ...
		// a generic ctor between _vecs would be nice, but maybe problematic for _mat = _sym
		//TEST_EQ(f.xxx(), float3(4,4,4));	
		TEST_EQ(fxxx.z, f.x);
		TEST_EQ(fxxx.y, f.y);
		TEST_EQ(fxxx.x, f.z);

//		TEST_EQ(f.xy(), float2(4,5));
//		TEST_EQ(f.yx(), float2(5,4));
//		TEST_EQ(f.yy(), float2(5,5));
	}
	
	/* more tests ...
	float2 float4
	int2 int3 int4
	
	default-template vectors of dif sizes (5 maybe? ... )
		assert that no .x exists to verify	
	
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
