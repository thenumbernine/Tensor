#include "Test/Test.h"

void test_Antisymmetric() {

	// antisymmetric matrix
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
