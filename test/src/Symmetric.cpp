#include "Test/Test.h"

//#define STORAGE_LOWER	//lower-triangular
#define STORAGE_UPPER	//upper-triangular

template<typename T>
void verifyAccessSym(T & a){
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
}

void test_Symmetric() {
	//symmetric
	
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

	verifyAccessSym<decltype(a)>(a);
	verifyAccessSym<decltype(a) const>(a);
	
	auto f = [](int i, int j) -> float { return i*i + j*j; };
	verifyAccessRank2<decltype(a)>(a, f);
	verifyAccessRank2<decltype(a) const>(a, f);

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


#ifdef STORAGE_LOWER // lower-triangular
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(0), Tensor::int2(0,0));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(1), Tensor::int2(1,0));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(2), Tensor::int2(1,1));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(3), Tensor::int2(2,0));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(4), Tensor::int2(2,1));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(5), Tensor::int2(2,2));
#endif                                                                       
#ifdef STORAGE_UPPER // upper-triangular                                     
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(0), Tensor::int2(0,0));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(1), Tensor::int2(0,1));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(2), Tensor::int2(1,1));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(3), Tensor::int2(0,2));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(4), Tensor::int2(1,2));
	TEST_EQ(Tensor::float3s3::getLocalReadForWriteIndex(5), Tensor::int2(2,2));
#endif

	for (int i = 0; i < Tensor::float3s3::localCount; ++i) {
		std::cout << i << "\t" << Tensor::float3s3::getLocalReadForWriteIndex(i) << std::endl;
	}
	
	// this is symmetric, it shouldn't matter
	std::cout << "getLocalWriteForReadIndex" << std::endl;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			std::cout << "\t" << Tensor::float3s3::getLocalWriteForReadIndex(i,j);
		}
		std::cout << std::endl;
	}

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
#ifdef STORAGE_LOWER // lower triangular
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
#endif // upper triangular
#ifdef STORAGE_UPPER	
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
