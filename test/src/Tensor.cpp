#include "Tensor/Tensor.h"
#include "Tensor/Inverse.h"
#include "Common/Test.h"

// TODO test everything a second time but with const access

// static tests here:

void test_Tensor() {
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
		TEST_EQ(f.dims, 3);
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

		// outer product
		// hmm, in the old days macros couldn't detect <>'s so you'd have to wrap them in ()'s if the <>'s had ,'s in them
		// now same persists for {}'s it seems
		auto fouterg = outer(f,g);
		TEST_EQ(fouterg, Tensor::float3x3(
			{28, 4, 8},
			{35, 5, 10},
			{49, 7, 14}
		));

		auto gouterf = transpose(fouterg);
		TEST_EQ(gouterf, Tensor::float3x3(
			{28, 35, 49},
			{4, 5, 7},
			{8, 10, 14}
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

	//old libraries' tests
	{
		{
			//arg ctor works
			Tensor::float3 a(1,2,3);
			
			//bracket ctor works
			Tensor::float3 b = {4,5,6};

			//access
			TEST_EQ(a(0), 1);
			TEST_EQ(a[0], 1);

			//make sure GenericArray functionality works
			TEST_EQ(Tensor::float3(1), Tensor::float3(1,1,1));
			
			// new lib doesn't support this ... but should it?
			//TEST_EQ(Tensor::float3(1,2), Tensor::float3(1,2,0));
			
			TEST_EQ(b + a, Tensor::float3(5,7,9));
			TEST_EQ(b - a, Tensor::float3(3,3,3));
			TEST_EQ(b * a, Tensor::float3(4, 10, 18));
			TEST_EQ(Tensor::float3(2,4,6)/Tensor::float3(1,2,3), Tensor::float3(2,2,2));
			TEST_EQ(b * 2., Tensor::float3(8, 10, 12));
			TEST_EQ(Tensor::float3(2,4,6)/2., Tensor::float3(1,2,3));	
		}
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

		//dims and rank.  really these are static_assert's, except dims, but it could be, but I'd have to constexpr some things ...
		TEST_EQ(m.rank, 2);
		TEST_EQ(m.dims, Tensor::int2(3,3));
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

		b = Tensor::float3s3{0,1,2,3,4,5};
		
		auto verifyAccess = []<typename T>(T & b){
			// testing (int,int) access
			TEST_EQ(b(0,0), 0);
			TEST_EQ(b(0,1), 1);
			TEST_EQ(b(0,2), 3);
			TEST_EQ(b(1,0), 1);
			TEST_EQ(b(1,1), 2);
			TEST_EQ(b(1,2), 4);
			TEST_EQ(b(2,0), 3);
			TEST_EQ(b(2,1), 4);
			TEST_EQ(b(2,2), 5);
			
			// testing fields 
			TEST_EQ(b.x_x, 0);
			TEST_EQ(b.x_y, 1);
			TEST_EQ(b.x_z, 3);
			TEST_EQ(b.y_x, 1);
			TEST_EQ(b.y_y, 2);
			TEST_EQ(b.y_z, 4);
			TEST_EQ(b.z_x, 3);
			TEST_EQ(b.z_y, 4);
			TEST_EQ(b.z_z, 5);

			// [][] access
			TEST_EQ(b[0][0], 0);
			TEST_EQ(b[0][1], 1);
			TEST_EQ(b[0][2], 3);
			TEST_EQ(b[1][0], 1);
			TEST_EQ(b[1][1], 2);
			TEST_EQ(b[1][2], 4);
			TEST_EQ(b[2][0], 3);
			TEST_EQ(b[2][1], 4);
			TEST_EQ(b[2][2], 5);

			// ()() access
			TEST_EQ(b(0)(0), 0);
			TEST_EQ(b(0)(1), 1);
			TEST_EQ(b(0)(2), 3);
			TEST_EQ(b(1)(0), 1);
			TEST_EQ(b(1)(1), 2);
			TEST_EQ(b(1)(2), 4);
			TEST_EQ(b(2)(0), 3);
			TEST_EQ(b(2)(1), 4);
			TEST_EQ(b(2)(2), 5);
		};
		verifyAccess.template operator()<decltype(b)>(b);
		verifyAccess.template operator()<decltype(b) const>(b);
		
		/*
		storing matrix 
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
		b = Tensor::float3s3([](int i, int j) -> float {
			return 3 * i + j;
		});
#if 1 // upper triangular
		// test storage order
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
		TEST_EQ(b.s[0], 0);	// xx
		TEST_EQ(b.s[1], 1); // xy
		TEST_EQ(b.s[2], 4); // yy
		TEST_EQ(b.s[3], 2); // xz
		TEST_EQ(b.s[4], 5); // yz
		TEST_EQ(b.s[5], 8); // zz
		// test arg ctor 
		TEST_EQ(b, Tensor::float3s3(0,1,4,2,5,8));
#endif


		// test symmetry
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
		
		TEST_EQ(a + 0.f, a);
		TEST_EQ(a * 1.f, a);
		TEST_EQ(a * 0.f, decltype(a)());
	
		// TODO how do GLSL matrix ctor from scalars work? 
		// do they initialize to full scalars like vecs do?
		// do they initialize to ident times scalar like math do?
	}

	// antisymmetric matrix
	{
		/*
		[ 0  1  2]
		[-1  0  3]
		[-2 -3  0]
		*/
		auto f = Tensor::float3a3{
			/*x_y=*/1,
			/*x_z=*/2,
			/*y_z=*/3
		};
		// (int,int) access
		f(0,0) = 1; //cannot write to diagonals
		f(1,1) = 2;
		f(2,2) = 3;

		auto verifyAccess = []<typename T>(T & f){
			TEST_EQ(f(0,0), 0);
			TEST_EQ(f(0,1), 1);
			TEST_EQ(f(0,2), 2);
			TEST_EQ(f(1,0), -1);
			TEST_EQ(f(1,1), 0);
			TEST_EQ(f(1,2), 3);
			TEST_EQ(f(2,0), -2);
			TEST_EQ(f(2,1), -3);
			TEST_EQ(f(2,2), 0);
			
			// "field" method access
			TEST_EQ(f.x_x(), 0);
			TEST_EQ(f.x_y(), 1);
			TEST_EQ(f.x_z(), 2);
			TEST_EQ(f.y_x(), -1);
			TEST_EQ(f.y_y(), 0);
			TEST_EQ(f.y_z(), 3);
			TEST_EQ(f.z_x(), -2);
			TEST_EQ(f.z_y(), -3);
			TEST_EQ(f.z_z(), 0);

			// [][] access
			TEST_EQ(f[0][0], 0);
			TEST_EQ(f[0][1], 1);
			TEST_EQ(f[0][2], 2);
			TEST_EQ(f[1][0], -1);
			TEST_EQ(f[1][1], 0);
			TEST_EQ(f[1][2], 3);
			TEST_EQ(f[2][0], -2);
			TEST_EQ(f[2][1], -3);
			TEST_EQ(f[2][2], 0);

			// ()() access
			TEST_EQ(f(0)(0), 0);
			TEST_EQ(f(0)(1), 1);
			TEST_EQ(f(0)(2), 2);
			TEST_EQ(f(1)(0), -1);
			TEST_EQ(f(1)(1), 0);
			TEST_EQ(f(1)(2), 3);
			TEST_EQ(f(2)(0), -2);
			TEST_EQ(f(2)(1), -3);
			TEST_EQ(f(2)(2), 0);
		};
		verifyAccess.template operator()<decltype(f)>(f);
		verifyAccess.template operator()<decltype(f) const>(f);

		// verify antisymmetric writes work
		for (int i = 0; i < f.dim<0>; ++i) {
			for (int j = 0; j < f.dim<1>; ++j) {
				float k = 1 + i + j;
				f(i,j) = k;
				if (i != j) {
					TEST_EQ(f(i,j), k);
					TEST_EQ(f(j,i), -k);
				} else {
					TEST_EQ(f(i,j), 0);
				}
			}
		}

		// TODO verify that 'float3a3::ExpandStorage<0> == float3x3' & same with <1> 

		// verify assignment to expanded type
		// TODO won't work until you get intN dereference in _asym
		//Tensor::float3x3 b = f;
		//ECHO(b);
	}

	// tensor with intermixed non-vec types:
	// vector-of-symmetric
	{
		//this is a T_ijk = T_ikj, i spans 3 dims, j and k span 2 dims
		using T2S3 = Tensor::_tensori<float, Tensor::index_vec<2>, Tensor::index_sym<3>>;
		
		// list ctor
		T2S3 t = {
			{1,2,3,4,5,6}, //x_x x_y y_y x_z y_z z_z
			{7,8,9,0,1,2},
		};
		// xyz field access
		TEST_EQ(t.x.x_x, 1);
		TEST_EQ(t.x.x_y, 2);
		TEST_EQ(t.x.y_y, 3);
		TEST_EQ(t.x.x_z, 4);
		TEST_EQ(t.x.y_z, 5);
		TEST_EQ(t.x.z_z, 6);
		TEST_EQ(t.y.x_x, 7);
		TEST_EQ(t.y.x_y, 8);
		TEST_EQ(t.y.y_y, 9);
		TEST_EQ(t.y.x_z, 0);
		TEST_EQ(t.y.y_z, 1);
		TEST_EQ(t.y.z_z, 2);
	
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

		// operators
		TEST_EQ(t + 0.f, t);
		TEST_EQ(t * 1.f, t);
		TEST_EQ(t * 0.f, decltype(t)());
	}
	{
		using T3S3 = Tensor::_tensori<float, Tensor::index_vec<3>, Tensor::index_sym<3>>;
		auto t = T3S3([](int i, int j, int k) -> float { return i+j+k; });
		auto verifyAccess = []<typename T>(T & t){
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					for (int k = 0; k < T::template dim<2>; ++k) {
						float e=i+j+k;
					
						//()()() and any possible merged ()'s
						TEST_EQ(t(i)(j)(k), e);
						TEST_EQ(t(i)(j,k), e);
						//_vec's operator()(int...) returns Scalar&
						//  but vec-of-sym needs to return a sym-Accessor object
						//TEST_EQ(t(i,j)(k), e);
						TEST_EQ(t(i,j,k), e);
						//[]()() ...
						TEST_EQ(t[i](j)(k), e);
						TEST_EQ(t[i](j,k), e);
						//()[]()
						TEST_EQ(t(i)[j](k), e);
						//()()[]
						TEST_EQ(t(i)(j)[k], e);
						//TEST_EQ(t(i,j)[k], e); // same problem as t(i,j)(k)
						// [][]() []()[] ()[][] [][][]
						TEST_EQ(t[i][j](k), e);
						TEST_EQ(t[i](j)[k], e);
						TEST_EQ(t(i)[j][k], e);
						TEST_EQ(t[i][j][k], e);
					}
				}
			}
		};
		verifyAccess.template operator()<decltype(t)>(t);
		verifyAccess.template operator()<decltype(t) const>(t);
	}
	
	// symmetric-of-vector
	{
		using TS33 = Tensor::_tensori<float, Tensor::index_sym<3>, Tensor::index_vec<3>>;
		auto t = TS33([](int i, int j, int k) -> float { return i+j+k; });
		auto verifyAccess = []<typename T>(T & t){
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					for (int k = 0; k < T::template dim<2>; ++k) {
						float e=i+j+k;
						//()()()
						TEST_EQ(t(i)(j)(k), e);
						TEST_EQ(t(i)(j,k), e);
						TEST_EQ(t(i,j)(k), e);
						TEST_EQ(t(i,j,k), e);
						//[]()() ...
						TEST_EQ(t[i](j)(k), e);
						TEST_EQ(t[i](j,k), e);
						//()[]()
						TEST_EQ(t(i)[j](k), e);
						//()()[]
						TEST_EQ(t(i)(j)[k], e);
						TEST_EQ(t(i,j)[k], e);
						// [][]() []()[] ()[][] [][][]
						TEST_EQ(t[i][j](k), e);
						TEST_EQ(t[i](j)[k], e);
						TEST_EQ(t(i)[j][k], e);
						TEST_EQ(t[i][j][k], e);
					}
				}
			}
		};
		verifyAccess.template operator()<decltype(t)>(t);
		verifyAccess.template operator()<decltype(t) const>(t);
	}

	// symmetric-of-symmetric
	{
		using TS3S3 = Tensor::_tensori<float, Tensor::index_sym<3>, Tensor::index_sym<3>>;
		auto t = TS3S3([](int i, int j, int k, int l) -> float { return i+j+k+l; });
		auto verifyAccess = []<typename T>(T & t){
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					for (int k = 0; k < T::template dim<2>; ++k) {
						for (int l = 0; l < T::template dim<3>; ++l) {
							float e =i+j+k+l;
							TEST_EQ(t(i)(j)(k)(l), e);
							TEST_EQ(t(i,j)(k)(l), e);
							TEST_EQ(t(i,j)(k,l), e);
							// TODO replace _sym's Accessor's TENSOR_ADD_RANK1_CALL_INDEX with stuff that returns objects instead of object-refs
							//TEST_EQ(t(i)(j,k)(l), e);
							TEST_EQ(t(i)(j)(k,l), e);
							TEST_EQ(t(i)(j,k,l), e);
							//TEST_EQ(t(i,j,k)(l), e);
							TEST_EQ(t(i,j,k,l), e);

							TEST_EQ(t[i](j)(k)(l), e);
							//TEST_EQ(t[i](j,k)(l), e);
							TEST_EQ(t[i](j)(k,l), e);
							TEST_EQ(t[i](j,k,l), e);
							
							TEST_EQ(t(i)[j](k)(l), e);
							TEST_EQ(t(i)[j](k,l), e);
							
							TEST_EQ(t(i)(j)[k](l), e);
							TEST_EQ(t(i,j)[k](l), e);
							
							TEST_EQ(t(i)(j)(k)[l], e);
							TEST_EQ(t(i,j)(k)[l], e);
							//TEST_EQ(t(i)(j,k)[l], e);
							TEST_EQ(t(i,j)(k)[l], e);
							
							TEST_EQ(t[i][j](k)(l), e);
							TEST_EQ(t[i][j](k,l), e);
							
							TEST_EQ(t[i](j)[k](l), e);
							
							TEST_EQ(t[i](j)(k)[l], e);
							//TEST_EQ(t[i](j,k)[l], e);
							
							TEST_EQ(t(i)[j][k](l), e);
							
							TEST_EQ(t(i)[j](k)[l], e);
							
							TEST_EQ(t(i)(j)[k][l], e);
							TEST_EQ(t(i,j)[k][l], e);
							
							TEST_EQ(t[i][j][k](l), e);
							TEST_EQ(t[i][j](k)[l], e);
							TEST_EQ(t[i](j)[k][l], e);
							TEST_EQ(t(i)[j][k][l], e);
							
							TEST_EQ(t[i][j][k][l], e);
						}
					}
				}
			}
		};
		verifyAccess.template operator()<decltype(t)>(t);
		verifyAccess.template operator()<decltype(t) const>(t);
	}

	// TODO antisymmetric of vector
	{
	}

	//antisymmetric of antisymmetric
	{
		using Real = double;
		using Riemann2 = Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<2>>;
		//using Riemann2 = Tensor::_asym<Tensor::_asym<Real, 2>, 2>;	// R_[ij][kl]
		//using Riemann2 = Tensor::_sym<Tensor::_asym<Real, 2>, 2>;	// ... R_(ij)[kl] ... 
		// how would I define R_( [ij] [kl ) ... i.e. R_ijkl = R_klij and R_ijkl = -R_jikl ?
		auto r = Riemann2{{1}};
		static_assert(Riemann2::rank == 4);
		static_assert(Riemann2::dim<0> == 2);
		static_assert(Riemann2::dim<1> == 2);
		static_assert(Riemann2::dim<2> == 2);
		static_assert(Riemann2::dim<3> == 2);
		static_assert(Riemann2::numNestings == 2);
		static_assert(Riemann2::count<0> == 1);
		static_assert(Riemann2::count<1> == 1);
		static_assert(sizeof(Riemann2) == sizeof(Real));
		auto r00 = r(0,0);	// this type will be a ZERO AntiSymRef wrapper around ... nothing ... 
		ECHO(r00);
		TEST_EQ(r00, (Tensor::AntiSymRef<Tensor::_asym<Real, 2>>()));	// r(0,0) is this type
		TEST_EQ(r00, (Tensor::_asym<Real, 2>{}));	// ... and r(0,0)'s operator== accepts its wrapped type
		TEST_EQ(r00(0,0), (Tensor::AntiSymRef<Real>()));	// r(0,0)(0,0) is this
		TEST_EQ(r00(0,0).how, Tensor::AntiSymRefHow::ZERO);
		TEST_EQ(r00(0,0), 0.);
		TEST_EQ(r00(0,1), 0.);
		TEST_EQ(r00(1,0), 0.);
		TEST_EQ(r00(1,1), 0.);
		auto r01 = r(0,1);	// this will point to the positive r.x_y element
		TEST_EQ(r01, (Tensor::_asym<Real, 2>{1}));
		TEST_EQ(r01(0,0), 0);	//why would this get a bad ref?
		TEST_EQ(r01(0,1), 1);
		TEST_EQ(r01(1,0), -1);
		TEST_EQ(r01(1,1), 0);
		auto r10 = r(1,0);
		TEST_EQ(r10, (Tensor::_asym<Real, 2>{-1}));
		TEST_EQ(r10(0,0), 0);
		TEST_EQ(r10(0,1), -1);
		TEST_EQ(r10(1,0), 1);
		TEST_EQ(r10(1,1), 0);
		auto r11 = r(1,1);
		TEST_EQ(r11(0,0), 0.);
		TEST_EQ(r11(0,1), 0.);
		TEST_EQ(r11(1,0), 0.);
		TEST_EQ(r11(1,1), 0.);
	}
	{
		constexpr int N = 3;
		using Riemann3 = Tensor::_tensori<double, Tensor::index_asym<N>, Tensor::index_asym<N>>;
		auto r = Riemann3();
		static_assert(Riemann3::rank == 4);
		static_assert(Riemann3::dim<0> == N);
		static_assert(Riemann3::dim<1> == N);
		static_assert(Riemann3::dim<2> == N);
		static_assert(Riemann3::dim<3> == N);
		static_assert(Riemann3::numNestings == 2);
		static_assert(Riemann3::count<0> == 3);	//3x3 antisymmetric has 3 unique components
		static_assert(Riemann3::count<1> == 3);
		//TODO some future work: R_ijkl = R_klij, so it's also symmetri between 1&2 and 3&4 ...
		// ... and optimizing for those should put us at only 6 unique values instead of 9
		static_assert(sizeof(Riemann3) == sizeof(double) * 9);

		double e = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < i; ++j) {
				for (int k = 0; k < N; ++k) {
					for (int l = 0; l < N; ++l) {
						r(i,j)(k,l) = ++e;
						if (i == j || k == l) {
							TEST_EQ(r(i,j)(k,l), 0);
						} else {
							TEST_EQ(r(i,j)(k,l), e);
							TEST_EQ(r(i,j)(l,k), -e);
							TEST_EQ(r(j,i)(k,l), -e);
							TEST_EQ(r(j,i)(l,k), e);
							
							TEST_EQ(r(i,j,k,l), e);
							TEST_EQ(r(i,j,l,k), -e);
							TEST_EQ(r(j,i,k,l), -e);
							TEST_EQ(r(j,i,l,k), e);
						}
					}
				}
			}
		}

// TODO change tensor generic ctor to (requires tensors  and) accept any type, iterate over write elements, assign one by one.

//		auto m = Tensor::ExpandAllIndexes<Riemann3>([&](Tensor::int4 i) -> double {
//			return r(i);
//		});
	}

	{
		// TODO verify 3- nestings deep of antisym works
	}

	// old libraries' tests
	{
		using Real = double;
		using Vector = Tensor::_tensor<Real,3>;
		
		Vector v = {1,2,3};
		TEST_EQ(v, Tensor::double3(1,2,3));
		
		using Metric = Tensor::_tensori<Real,Tensor::index_sym<3>>;
		Metric g;
		for (int i = 0; i < 3; ++i) {
			g(i,i) = 1;
		}
		TEST_EQ(g, Metric(1,0,1,0,0,1));

		using Matrix = Tensor::_tensor<Real,3,3>;
		Matrix h;
		int index = 0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				h(i,j) = ++index;
			}
		}
		TEST_EQ(h, (Matrix{{1,2,3},{4,5,6},{7,8,9}}));

		//iterator access
		int j = 0;
		Tensor::_tensor<Real, 3,3,3> ta;
		for (auto i = ta.begin(); i != ta.end(); ++i) {
			*i = j++;
		}
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				for (int k = 0; k < 3; ++k) {
					TEST_EQ(ta(i,j,k), i + 3 * (j + 3 * k));
				}
			}
		}

		//subtensor access not working
		Tensor::_tensor<Real,3,3> tb;
		for (auto i = tb.begin(); i != tb.end(); ++i) *i = 2.f;
		TEST_EQ(tb, Matrix(2.f));
		ta(0) = tb;
		TEST_EQ(ta, (Tensor::_tensor<Real,3,3,3>{{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}, {{1, 10, 19}, {4, 13, 22}, {7, 16, 25}}, {{2, 11, 20}, {5, 14, 23}, {8, 17, 26}}} ));
		Tensor::_tensor<Real, 3> tc;
		for (auto i = tc.begin(); i != tc.end(); ++i) *i = 3.;
		TEST_EQ(Tensor::double3(3), tc);
		ta(0,0) = tc;
		TEST_EQ(ta, (Tensor::_tensor<Real,3,3,3>{{{3, 3, 3}, {2, 2, 2}, {2, 2, 2}}, {{1, 10, 19}, {4, 13, 22}, {7, 16, 25}}, {{2, 11, 20}, {5, 14, 23}, {8, 17, 26}}}));

		//inverse
		Matrix m;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				m(i,j) = i == j ? 1 : 0;
			}
		}

		TEST_EQ(m, (Matrix{{1,0,0},{0,1,0},{0,0,1}}));
		TEST_EQ(Tensor::determinant(m), 1);
	}

	// more old tests
	// TODO these are static_assert's
	{
		using Real = double;
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<3>>::rank)== 1);
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<3>>::dim<0>)== 3);

		static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>>::rank)== 1);
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>>::dim<0>)== 4);

		static_assert((Tensor::_tensori<Real, Tensor::index_sym<3>>::rank)== 2);
		static_assert((Tensor::_tensori<Real, Tensor::index_sym<3>>::dim<0>)== 3);
		static_assert((Tensor::_tensori<Real, Tensor::index_sym<3>>::dim<1>)== 3);

		static_assert((Tensor::_tensori<Real, Tensor::index_vec<5>, Tensor::index_vec<6>>::rank)== 2);
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<5>, Tensor::index_vec<6>>::dim<0>)== 5);
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<5>, Tensor::index_vec<6>>::dim<1>)== 6);

		static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::rank)== 3);
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::dim<0>)== 4);
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::dim<1>)== 3);
		static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::dim<2>)== 3);

		static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::rank)== 4);
		static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<0>)== 2);
		static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<1>)== 2);
		static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<2>)== 3);
		static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<3>)== 3);
	}

	// verify that the outer of a vector and a sym is just that
	{
		auto a = Tensor::float2(2,3);
		//ECHO(a);
		static_assert(a.numNestings == 1);
		static_assert(a.count<0> == 2);
		static_assert(a.rank == 1);
		static_assert(a.dim<0> == 2);
		auto b = Tensor::float3s3(6,5,4,3,2,1);
		//ECHO(b);
		static_assert(b.numNestings == 1);
		static_assert(b.count<0> == 6);
		static_assert(b.rank == 2);
		static_assert(b.dim<0> == 3);
		static_assert(b.dim<1> == 3);
		auto c = outer(a,b);
		//ECHO(c);
		static_assert(c.numNestings == 2);
		static_assert(c.count<0> == a.count<0>);
		static_assert(c.count<1> == b.count<0>);
		static_assert(c.rank == 3);
		static_assert(c.dim<0> == 2);
		static_assert(c.dim<1> == 3);
		static_assert(c.dim<2> == 3);
		auto d = outer(b,a);
		//ECHO(d);
		static_assert(d.numNestings == 2);
		static_assert(d.count<0> == b.count<0>);
		static_assert(d.count<1> == a.count<0>);
		static_assert(d.rank == 3);
		static_assert(d.dim<0> == 3);
		static_assert(d.dim<1> == 3);
		static_assert(d.dim<2> == 2);
	}
}
