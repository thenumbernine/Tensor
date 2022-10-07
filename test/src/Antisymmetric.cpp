#include "Test/Test.h"

//#define STORAGE_LOWER	//lower-triangular
#define STORAGE_UPPER	//upper-triangular

template<typename T, typename F>
void verifyAccessAntisym(T & t, F f) {
	// "field" method access
	TEST_EQ(t.x_x(), f(0,0));
	TEST_EQ(t.x_y(), f(0,1));
	TEST_EQ(t.x_z(), f(0,2));
	TEST_EQ(t.y_x(), f(1,0));
	TEST_EQ(t.y_y(), f(1,1));
	TEST_EQ(t.y_z(), f(1,2));
	TEST_EQ(t.z_x(), f(2,0));
	TEST_EQ(t.z_y(), f(2,1));
	TEST_EQ(t.z_z(), f(2,2));
}

void test_Antisymmetric() {

#ifdef STORAGE_LOWER // lower-triangular
	TEST_EQ(Tensor::float3a3::getLocalReadForWriteIndex(0), Tensor::int2(1,0));
	TEST_EQ(Tensor::float3a3::getLocalReadForWriteIndex(1), Tensor::int2(2,0));
	TEST_EQ(Tensor::float3a3::getLocalReadForWriteIndex(2), Tensor::int2(2,1));
#endif                                                                       
#ifdef STORAGE_UPPER // upper-triangular                                     
	TEST_EQ(Tensor::float3a3::getLocalReadForWriteIndex(0), Tensor::int2(0,1));
	TEST_EQ(Tensor::float3a3::getLocalReadForWriteIndex(1), Tensor::int2(0,2));
	TEST_EQ(Tensor::float3a3::getLocalReadForWriteIndex(2), Tensor::int2(1,2));
#endif

	// antisymmetric matrix
	/*
	[ 0  1  2]
	[-1  0  3]
	[-2 -3  0]
	*/
	auto f = [](int i, int j) -> float { return sign(j-i)*(i+j); };
#if 0 // list initializer with lambda is failing	
	auto t = Tensor::float3a3{
		/*x_y=*/f(0,1),
		/*x_z=*/f(0,2),
		/*y_z=*/f(1,2)
	};
#endif
#if 0 //same	
	auto t = Tensor::float3a3{
		/*x_y=*/1,
		/*x_z=*/2,
		/*y_z=*/3
	};
#endif
#if 1
	auto t = Tensor::float3a3(1, 2, 3);
#endif
#ifdef STORAGE_UPPER
	TEST_EQ(t.s[0], 1);
	TEST_EQ(t.s[1], 2);
	TEST_EQ(t.s[2], 3);
#endif
#ifdef STORAGE_LOWER
	TEST_EQ(t.s[0], -1);
	TEST_EQ(t.s[1], -2);
	TEST_EQ(t.s[2], -3);
#endif
	ECHO(t);
	TEST_EQ(t(0,0), 0);
	TEST_EQ(t(0,1), 1);
	TEST_EQ(t(0,2), 2);
	TEST_EQ(t(1,0), -1);
	TEST_EQ(t(1,1), 0);
	TEST_EQ(t(1,2), 3);
	TEST_EQ(t(2,0), -2);
	TEST_EQ(t(2,1), -3);
	TEST_EQ(t(2,2), 0);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			std::cout << "\t" << t(i,j);
		}
		std::cout << std::endl;
	}

	// (int,int) access
	t(0,0) = 1; //cannot write to diagonals
	TEST_EQ(t(0,0),0);
	t(1,1) = 2;
	TEST_EQ(t(1,1),0);
	t(2,2) = 3;
	TEST_EQ(t(2,2),0);

	auto a = Tensor::float3a3(f);
	ECHO(a);
	TEST_EQ(a, t);

	//verify ctor from lambda for int2
	auto f2 = [](Tensor::int2 ij) -> float { return ij.x + ij.y; };
	auto b = Tensor::float3a3(f2);
	ECHO(b);
	TEST_EQ(b, t);

	verifyAccessAntisym<decltype(t)>(t, f);
	verifyAccessAntisym<decltype(t) const>(t, f);
	
	verifyAccessRank2<decltype(t)>(t, f);
	verifyAccessRank2<decltype(t) const>(t, f);

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

	{
		Tensor::float3x3 m = Tensor::float3x3{
			{0,1,2},
			{-1,0,3},
			{-2,-3,0}
		};
		ECHO(m);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cout << "\t" << m(i,j);
			}
			std::cout << std::endl;
		}
		Tensor::float3a3 as = m;
		ECHO(as);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cout << "\t" << as(i,j);
			}
			std::cout << std::endl;
		}
		//does assigning to _mat work?
		Tensor::float3x3 mas = as;
		ECHO(mas);
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cout << "\t" << mas(i,j);
			}
			std::cout << std::endl;
		}
		// does equality between _asym and assigned _mat work?
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				TEST_EQ(as(i,j), m(i,j));
			}
		}
		TEST_EQ(as,m);
	}
}
