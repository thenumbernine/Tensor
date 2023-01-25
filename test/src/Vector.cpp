#include "Test/Test.h"

namespace Tensor {

#define TEST_TENSOR_ADD_VECTOR_STATIC_ASSERTS(nick,ctype,dim1)\
static_assert(sizeof(nick##dim1) == sizeof(ctype) * dim1);\
static_assert(std::is_same_v<nick##dim1::Scalar, ctype>);\
static_assert(std::is_same_v<nick##dim1::Inner, ctype>);\
static_assert(nick##dim1::rank == 1);\
static_assert(nick##dim1::dim<0> == dim1);\
static_assert(nick##dim1::numNestings == 1);\
static_assert(nick##dim1::count<0> == dim1);

#define TEST_TENSOR_ADD_MATRIX_STATIC_ASSERTS(nick, ctype, dim1, dim2)\
static_assert(sizeof(nick##dim1##x##dim2) == sizeof(ctype) * dim1 * dim2);\
static_assert(nick##dim1##x##dim2::rank == 2);\
static_assert(nick##dim1##x##dim2::dim<0> == dim1);\
static_assert(nick##dim1##x##dim2::dim<1> == dim2);\
static_assert(nick##dim1##x##dim2::numNestings == 2);\
static_assert(nick##dim1##x##dim2::count<0> == dim1);\
static_assert(nick##dim1##x##dim2::count<1> == dim2);

#define TEST_TENSOR_ADD_SYMMETRIC_STATIC_ASSERTS(nick, ctype, dim12)\
static_assert(sizeof(nick##dim12##s##dim12) == sizeof(ctype) * triangleSize(dim12));\
static_assert(std::is_same_v<typename nick##dim12##s##dim12::Scalar, ctype>);\
static_assert(nick##dim12##s##dim12::rank == 2);\
static_assert(nick##dim12##s##dim12::dim<0> == dim12);\
static_assert(nick##dim12##s##dim12::dim<1> == dim12);\
static_assert(nick##dim12##s##dim12::numNestings == 1);\
static_assert(nick##dim12##s##dim12::count<0> == triangleSize(dim12));

#define TEST_TENSOR_ADD_ANTISYMMETRIC_STATIC_ASSERTS(nick, ctype, dim12)\
static_assert(sizeof(nick##dim12##a##dim12) == sizeof(ctype) * triangleSize(dim12-1));\
static_assert(std::is_same_v<typename nick##dim12##a##dim12::Scalar, ctype>);\
static_assert(nick##dim12##a##dim12::rank == 2);\
static_assert(nick##dim12##a##dim12::dim<0> == dim12);\
static_assert(nick##dim12##a##dim12::dim<1> == dim12);\
static_assert(nick##dim12##a##dim12::numNestings == 1);\
static_assert(nick##dim12##a##dim12::count<0> == triangleSize(dim12-1));

#define TEST_TENSOR_ADD_IDENTITY_STATIC_ASSERTS(nick, ctype, dim12)\
static_assert(sizeof(nick##dim12##i##dim12) == sizeof(ctype));\
static_assert(std::is_same_v<typename nick##dim12##i##dim12::Scalar, ctype>);\
static_assert(nick##dim12##i##dim12::rank == 2);\
static_assert(nick##dim12##i##dim12::dim<0> == dim12);\
static_assert(nick##dim12##i##dim12::dim<1> == dim12);\
static_assert(nick##dim12##i##dim12::numNestings == 1);\
static_assert(nick##dim12##i##dim12::count<0> == 1);

#define TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_STATIC_ASSERTS(nick, ctype, localDim, localRank, suffix)\
static_assert(sizeof(nick##suffix) == sizeof(ctype) * consteval_symmetricSize(localDim, localRank));\
static_assert(std::is_same_v<typename nick##suffix::Scalar, ctype>);\
static_assert(nick##suffix::rank == localRank);\
static_assert(nick##suffix::dim<0> == localDim); /* TODO repeat depending on dimension */\
static_assert(nick##suffix::numNestings == 1);\
static_assert(nick##suffix::count<0> == consteval_symmetricSize(localDim, localRank));

#define TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_STATIC_ASSERTS(nick, ctype, localDim, localRank, suffix)\
static_assert(sizeof(nick##suffix) == sizeof(ctype) * consteval_antisymmetricSize(localDim, localRank));\
static_assert(std::is_same_v<typename nick##suffix::Scalar, ctype>);\
static_assert(nick##suffix::rank == localRank);\
static_assert(nick##suffix::dim<0> == localDim); /* TODO repeat depending on dimension */\
static_assert(nick##suffix::numNestings == 1);\
static_assert(nick##suffix::count<0> == consteval_antisymmetricSize(localDim, localRank));


#define TEST_TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, dim1)\
TEST_TENSOR_ADD_VECTOR_STATIC_ASSERTS(nick, ctype,dim1);

#define TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, dim1, dim2)\
TEST_TENSOR_ADD_MATRIX_STATIC_ASSERTS(nick, ctype, dim1, dim2)

#define TEST_TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, dim12)\
TEST_TENSOR_ADD_SYMMETRIC_STATIC_ASSERTS(nick, ctype, dim12)

#define TEST_TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, dim12)\
TEST_TENSOR_ADD_ANTISYMMETRIC_STATIC_ASSERTS(nick, ctype, dim12)

#define TEST_TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, dim12)\
TEST_TENSOR_ADD_IDENTITY_STATIC_ASSERTS(nick, ctype, dim12)

#define TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, localDim, localRank, suffix)\
TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_STATIC_ASSERTS(nick, ctype, localDim, localRank, suffix)

#define TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, localDim, localRank, suffix)\
TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_STATIC_ASSERTS(nick, ctype, localDim, localRank, suffix)

#define TEST_TENSOR_ADD_NICKNAME_TYPE(nick, ctype)\
/* typed vectors */\
TEST_TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, 2)\
TEST_TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, 3)\
TEST_TENSOR_ADD_VECTOR_NICKCNAME_TYPE_DIM(nick, ctype, 4)\
/* typed matrices */\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 2, 2)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 2, 3)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 2, 4)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 3, 2)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 3, 3)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 3, 4)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 4, 2)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 4, 3)\
TEST_TENSOR_ADD_MATRIX_NICKNAME_TYPE_DIM(nick, ctype, 4, 4)\
/* identity matrix */\
TEST_TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, 2)\
TEST_TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, 3)\
TEST_TENSOR_ADD_IDENTITY_NICKNAME_TYPE_DIM(nick, ctype, 4)\
/* typed symmetric matrices */\
TEST_TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2)\
TEST_TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3)\
TEST_TENSOR_ADD_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4)\
/* typed antisymmetric matrices */\
TEST_TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2)\
TEST_TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3)\
TEST_TENSOR_ADD_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4)\
/* totally symmetric tensors */\
TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 3, 2s2s2)\
TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 3, 3s3s3)\
TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 3, 4s4s4)\
TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 4, 2s2s2s2)\
TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 4, 3s3s3s3)\
TEST_TENSOR_ADD_TOTALLY_SYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 4, 4s4s4s4)\
/* totally antisymmetric tensors */\
/* can't exist: TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 3, 2a2a2)*/\
TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 3, 3a3a3)\
TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 3, 4a4a4)\
/* can't exist: TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 2, 4, 2a2a2a2)*/\
/* can't exist: TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 3, 4, 3a3a3a3)*/\
TEST_TENSOR_ADD_TOTALLY_ANTISYMMETRIC_NICKNAME_TYPE_DIM(nick, ctype, 4, 4, 4a4a4a4)

#define TEST_TENSOR_ADD_UTYPE(x)	TEST_TENSOR_ADD_NICKNAME_TYPE(u##x,unsigned x)

#define TEST_TENSOR_ADD_TYPE(x)	TEST_TENSOR_ADD_NICKNAME_TYPE(x,x)

TEST_TENSOR_ADD_TYPE(bool)
TEST_TENSOR_ADD_TYPE(char)
TEST_TENSOR_ADD_UTYPE(char)
TEST_TENSOR_ADD_TYPE(short)
TEST_TENSOR_ADD_UTYPE(short)
TEST_TENSOR_ADD_TYPE(int)
TEST_TENSOR_ADD_UTYPE(int)
TEST_TENSOR_ADD_TYPE(float)
TEST_TENSOR_ADD_TYPE(double)
TEST_TENSOR_ADD_NICKNAME_TYPE(size, size_t)
TEST_TENSOR_ADD_NICKNAME_TYPE(intptr, intptr_t)
TEST_TENSOR_ADD_NICKNAME_TYPE(uintptr, uintptr_t)
TEST_TENSOR_ADD_NICKNAME_TYPE(ldouble, long double)

}

namespace Tests {
	using namespace Tensor;
	using namespace Common;
	STATIC_ASSERT_EQ(int3::dim<0>, 3);
	STATIC_ASSERT_EQ(int3::rank, 1);
	STATIC_ASSERT_EQ((seq_get_v<0, typename int3::dimseq>), 3);
	STATIC_ASSERT_EQ(int3::totalCount, 3);
}

void test_Vector() {
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
		
		// initializer list ctor
		Tensor::float3 g = {7,1,2};
		
		//.dims
		static_assert(f.rank == 1);
		static_assert(f.dims() == 3);
		static_assert(f.dim<0> == 3);
		static_assert(f.numNestings == 1);
		static_assert(f.count<0> == 3);
	
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

		// indexing
		{
			auto f = [](int i) -> float { return i+1; };
			Tensor::float3 t(f);
			verifyAccessRank1<decltype(t)>(t, f);
			verifyAccessRank1<decltype(t) const>(t, f);
		}

		//lambda ctor
		TEST_EQ(f, Tensor::float3([](int i) -> float { return 4 + i * (i + 1) / 2; }));
		TEST_EQ(f, Tensor::float3([](Tensor::intN<1> i) -> float { return 4 + i(0) * (i(0) + 1) / 2; }));

		// scalar ctor
		TEST_EQ(Tensor::float3(3), Tensor::float3(3,3,3));

		// casting
		Tensor::int3 fi = {4,5,7};
		Tensor::double3 fd = {4,5,7};
		TEST_EQ(f, (Tensor::float3)fi);
		TEST_EQ(f, (Tensor::float3)fd);

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
	
		// operators
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
		TEST_EQ(f/g, Tensor::float3(0.57142857142857, 5.0, 3.5));	// wow, this equality passes while sqrt(90) fails
		// unary
		TEST_EQ(-f, Tensor::float3(-4, -5, -7));

		// for vector*vector I'm picking the scalar-result of the 2 GLSL options (are there three? can you do mat = vec * vec in GLSL?)
		//  this fits with general compatability of tensor operator* being outer+contract
		TEST_EQ(f*g, 47)

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

		//operator<< works?
		std::cout << f << std::endl;
		// to_string works?
		std::cout << std::to_string(f) << std::endl;

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
		// a generic ctor between vecs would be nice, but maybe problematic for mat = sym
		TEST_EQ(Tensor::float3(f.zyx()), Tensor::float3(7,5,4));
		TEST_EQ(Tensor::float2(f.xy()), Tensor::float2(4,5));
		TEST_EQ(Tensor::float2(f.yx()), Tensor::float2(5,4));
		TEST_EQ(Tensor::float2(f.yy()), Tensor::float2(5,5));
		{
			auto x = Tensor::float3(1,2,3);
			x = x.yzx();
			TEST_EQ(x, Tensor::float3(2,3,1));
// ERROR: no matching constructor for initialization of 'std::reference_wrapper<float>' 
//			x.yzx() = Tensor::float3(7,8,9);
//			TEST_EQ(x, Tensor::float3(9,7,8));
		}

static_assert(sizeof(Tensor::float3a3a3) == sizeof(float));

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

		Tensor::vec<float,5> j = {5,6,7,8,9};
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

		//verify iterators look alright
		// btw std::copy might work dif in release than debug?
		{
			std::array<float, 3> fa;
			auto fai = fa.begin();
			auto fi = f.begin();
			TEST_EQ(fi.index, Tensor::intN<1>(0));
			TEST_NE(fi, f.end());
			TEST_EQ(&*fi, &f[0]);
			TEST_EQ(&*fai, &fa[0]);
			++fai; ++fi;
			TEST_EQ(fi.index, Tensor::intN<1>(1));
			TEST_NE(fi, f.end());
			TEST_EQ(&*fi, &f[1]);
			TEST_EQ(&*fai, &fa[1]);
			++fai; ++fi;
			TEST_EQ(fi.index, Tensor::intN<1>(2));
			TEST_NE(fi, f.end());
			TEST_EQ(&*fi, &f[2]);
			TEST_EQ(&*fai, &fa[2]);
			++fai; ++fi;
			TEST_EQ(fi.index, Tensor::intN<1>(3));
			TEST_EQ(fi, f.end());
		}

		// iterator copy from somewhere else
		{
			std::array<float, 3> fa;
			std::copy(f.begin(), f.end(), fa.begin());
			TEST_EQ(fa[0], f[0]);
			TEST_EQ(fa[1], f[1]);
			TEST_EQ(fa[2], f[2]);
		}

		// operators
		operatorScalarTest(f);
	
		// contract

		TEST_EQ((Tensor::contract<0,0>(Tensor::float3(1,2,3))), 6);
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
			TEST_EQ(b * a, 32);
			TEST_EQ(Tensor::float3(2,4,6)/Tensor::float3(1,2,3), Tensor::float3(2,2,2));
			TEST_EQ(b * 2., Tensor::float3(8, 10, 12));
			TEST_EQ(Tensor::float3(2,4,6)/2., Tensor::float3(1,2,3));
		}
	}

	//equivalent of tensor ctor of varying dimension
	{
		Tensor::float2 t = {1,2};
		Tensor::float3 x;
		auto w = x.write();
		for (auto i = w.begin(); i != w.end(); ++i) {
			/* TODO instead an index range iterator that spans the minimum of dims of this and t */
			if (Tensor::float2::validIndex(i.readIndex)) {
				/* If we use operator()(intN<>) access working for asym ... */
				/**i = (Scalar)t(i.readIndex);*/
				/* ... or just replace the internal storage with std::array ... */
				*i = std::apply(t, i.readIndex.s);
			} else {
				*i = decltype(x)::Scalar();
			}
		}
	}
	{
		using namespace Tensor;
		float2 a = {1,2};
		float3 b = a;
		TEST_EQ(b, float3(1,2,0));
	}
}
