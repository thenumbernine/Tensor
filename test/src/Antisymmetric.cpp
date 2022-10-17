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
	auto t = Tensor::float3a3{
		/*x_y=*/f(0,1),
		/*x_z=*/f(0,2),
		/*y_z=*/f(1,2)
	};
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


	{
		Tensor::float3x3 m = {
			{1,2,3},
			{4,5,6},
			{7,8,9},
		};
		Tensor::float3a3 a([](int i, int j) -> float { return sign(j-i)*(i+j); });

// works with unoptimized matrix-mul
// crashing with the optimized version of matrix-mul
		auto ma = m * a;
		static_assert(std::is_same_v<decltype(ma), Tensor::float3x3>);
		TEST_EQ(ma, (Tensor::float3x3{
			{-8, -8, 8},
			{-17, -14, 23},
			{-26, -20, 38},
		}));

		static_assert(std::is_same_v<decltype(makeSym(ma)), Tensor::float3s3>);
		TEST_EQ( .5f * (ma + transpose(ma)), (Tensor::float3x3)makeSym(ma));
		static_assert(std::is_same_v<decltype(ma.makeAsym()), Tensor::float3a3>);
		TEST_EQ( .5f * (ma - transpose(ma)), (Tensor::float3x3)ma.makeAsym());

		// TODO outer of antisym and ident is failing ...
		auto I = Tensor::_ident<float, 3>(1);
		auto aOuterI = outer(a, I);
		static_assert(std::is_same_v<decltype(aOuterI), Tensor::_tensori<float, Tensor::storage_asym<3>, Tensor::storage_ident<3>>>);
		static_assert(std::is_same_v<decltype(outer(I, a)), Tensor::_tensori<float, Tensor::storage_ident<3>, Tensor::storage_asym<3>>>);
		static_assert(sizeof(aOuterI) == sizeof(float) * 3); // expanded storage would be 3^4 = 81 floats, but this is just 3 ...

		auto aTimesI = a * I;
		// the matrix-mul will expand the antisymmetric matrix storage to a matrix
		static_assert(std::is_same_v<decltype(aTimesI), Tensor::float3x3>);
		// crashing
		TEST_EQ(aTimesI, a);
	}

	{
		auto a = Tensor::float3a3(1,2,3);
#if 1 //tensor ctor from Accessor?  shoud need Accessor .rank etc defined ...
		auto ax = a[0];
		//ECHO(ax);	// TODO operator<< for Accessor?
		// works
		auto axt = Tensor::float3(ax);
		ECHO(axt);
		// but I thought .rank and .dims needed to match?
		//ECHO(ax.rank);
		//ECHO(ax.dims());
		/* yeah, ax.rank doesn't exist,
		so what kind of ctor is being used for the conversion?
		oh wait, the Accessor might be usings its containing class' rank and dims for iterator as well
		that would mean ... meh
		looks like rank-2 Accessors only use
			TENSOR_ADD_RANK1_CALL_INDEX_AUX()\
			TENSOR_ADD_INT_VEC_CALL_INDEX()\
		... and neither uses 'This'
		... then rank-N Accessors use just 
			TENSOR_ADD_RANK1_CALL_INDEX_AUX()\
		so that means overall 'This' is not used by any Accessor
		and that means I can use 'This' within the Accessor to make it more compat with things.
		In fact I should move Accessors outside of the classes and just 'using' them in...
		ok now they're added
		*/
		ECHO(ax.rank);
		ECHO(ax.dims());
		ECHO(Tensor::lenSq(ax));
#endif
	}
	{
#if 1 // Accessor 's tensor member methods
		auto a = Tensor::float3a3(1,2,3);
		auto ax = a[0];
		ECHO(ax.lenSq());
		auto ay = a[1];
		ECHO(ay.lenSq());
		auto az = a[2];
		ECHO(az.lenSq());
#endif
	}
}
