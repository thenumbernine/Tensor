#include "Test/Test.h"

namespace MathTest {
	using namespace Tensor;
	using namespace std;
	static_assert(is_same_v<decltype(makeSym(_tensorr<float,3,1>())), _vec<float,3>>);
	static_assert(is_same_v<decltype(makeSym(_tensorr<float,3,2>())), _sym<float,3>>);
	static_assert(is_same_v<decltype(makeSym(_tensorr<float,3,3>())), _symR<float,3,3>>);
	static_assert(is_same_v<decltype(makeSym(_tensorr<float,3,4>())), _symR<float,3,4>>);
}

#define TENSOR_TEST_1(\
	funcName,\
	resultType, resultArgs,\
	inputAType, inputAArgs\
)\
	/* member */\
	/*  lvalues */\
	[](inputAType const & a){\
		auto c = a.funcName();\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs);\
	/*  rvalues */\
	[](inputAType && a){\
		auto c = a.funcName();\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs);\
	/* this should be rvalue too right? */\
	{\
		auto c = (inputAType inputAArgs).funcName();\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}\
	/* global */\
	/*  lvalues */\
	[](inputAType const & a){\
		auto c = funcName(a);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs);\
	/*  rvalues */\
	[](inputAType && a){\
		auto c = funcName(a);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs);\
	/* same? */\
	{\
		auto c = funcName(inputAType inputAArgs);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}

#define TENSOR_TEST_2(\
	funcName,\
	resultType, resultArgs,\
	inputAType, inputAArgs,\
	inputBType, inputBArgs\
)\
	/* member */\
	/*  lvalues */\
	[](inputAType const & a, inputBType const & b){\
		auto c = a.funcName(b);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs, inputBType inputBArgs);\
	/*  rvalues */\
	[](inputAType && a, inputBType && b){\
		auto c = a.funcName(b);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs, inputBType inputBArgs);\
	/*  same? */\
	{\
		auto c = (inputAType inputAArgs).funcName(inputBType inputBArgs);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}\
	/* global */\
	/*  lvalues */\
	[](inputAType const & a, inputBType const & b){\
		auto c = funcName(a,b);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs, inputBType inputBArgs);\
	/*  rvalues */\
	[](inputAType && a, inputBType && b){\
		auto c = funcName(a,b);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}(inputAType inputAArgs, inputBType inputBArgs);\
	/*  same? */\
	{\
		auto c = funcName(inputAType inputAArgs, inputBType inputBArgs);\
		static_assert(std::is_same_v<decltype(c), resultType>);\
		TEST_EQ(c, resultType resultArgs);\
	}

// testing the tensor math functions, hopefully as lvalues and rvalues, as globals and members ...
void test_Math() {
	using namespace Tensor;

	// TODO test dif ranks, test sym vs asym
	//  for some, test template args
	TENSOR_TEST_2(elemMul,			float3,		(4, 10, 18),	float3,	(1,2,3),	float3,	(4,5,6));
	TENSOR_TEST_2(matrixCompMult,	float3,		(4, 10, 18),	float3,	(1,2,3),	float3,	(4,5,6));
	TENSOR_TEST_2(hadamard,			float3,		(4, 10, 18),	float3,	(1,2,3),	float3,	(4,5,6));
	TENSOR_TEST_2(inner,			float,		(32),			float3,	(1,2,3),	float3,	(4,5,6));
	TENSOR_TEST_2(dot,				float,		(32),			float3,	(1,2,3),	float3,	(4,5,6));
	TENSOR_TEST_1(lenSq,			float,		(14),			float3,	(1,2,3));
	TENSOR_TEST_1(length,			float,		(43),			float3,	(9,18,38));
	TENSOR_TEST_2(distance,			float,		(43),			float3,	(2,27,33),	float3,	(-7,9,-5));
	TENSOR_TEST_1(normalize,		float3,		(0, .6f, .8f),	float3,	(0,3,4));
	TENSOR_TEST_2(cross,			float3,		(0,0,1),		float3,	(1,0,0),	float3,	(0,1,0));
	TENSOR_TEST_2(outer,			float3x3,	({{0,1,0},{0,0,0},{0,0,0}}),	float3, (1,0,0), float3, (0,1,0));
	TENSOR_TEST_2(outerProduct,		float3x3,	({{0,1,0},{0,0,0},{0,0,0}}),	float3, (1,0,0), float3, (0,1,0));
	//<m,n> which indexes to transpose
	// verify sym tr = sym, asym tr = -asym, symR and asymR too, ident tr = ident
	TENSOR_TEST_1(transpose,		float3x3,	({{1,4,7},{2,5,8},{3,6,9}}),	float3x3,	({{1,2,3},{4,5,6},{7,8,9}}));
	//<m,n> which indexes to contract / trace
	TENSOR_TEST_1(contract,			float,		(15.f),			float3x3,	({{1,4,7},{2,5,8},{3,6,9}}));
	TENSOR_TEST_1(trace,			float,		(15.f),			float3x3,	({{1,4,7},{2,5,8},{3,6,9}}));
	//<index,count> = which indexes & how many to contract
	TENSOR_TEST_1(contractN,		float,		(15.f),			float3x3,	({{1,4,7},{2,5,8},{3,6,9}}));
	//<num> = how many indexes to contract
	TENSOR_TEST_2(interior,			float,		(32),			float3,	(1,2,3),	float3,	(4,5,6));
	TENSOR_TEST_1(diagonal,			float3s3,	(float3x3{{1,0,0},{0,2,0},{0,0,3}}),	float3,	(1,2,3));
	//makeSym
	//makeAsym
	//wedge
	//hodgeDual
	//operator*
	TENSOR_TEST_1(determinant,		float,		(1.f),			float3x3,	({{1,0,0},{0,1,0},{0,0,1}}));
	TENSOR_TEST_1(inverse,			float3x3,	({{1,0,0},{0,1,0},{0,0,1}}),	float3x3,	({{1,0,0},{0,1,0},{0,0,1}}));
	TENSOR_TEST_2(inverse,			float3x3,	({{1,0,0},{0,1,0},{0,0,1}}),	float3x3,	({{1,0,0},{0,1,0},{0,0,1}}), float, (1.f));

	static_assert(sizeof(_tensori<float, storage_zero<3>, storage_zero<3>, storage_zero<3>>) == sizeof(float));
	
	static_assert(std::is_same_v<decltype(makeSym(float3a3())), _tensori<float, storage_zero<3>, storage_zero<3>>>);
	static_assert(std::is_same_v<decltype(makeSym(float3a3a3())), _tensori<float, storage_zero<3>, storage_zero<3>, storage_zero<3>>>);
	
	static_assert(std::is_same_v<decltype(makeAsym(float3s3())), _tensori<float, storage_zero<3>, storage_zero<3>>>);
	static_assert(std::is_same_v<decltype(makeAsym(float3s3s3())), _tensori<float, storage_zero<3>, storage_zero<3>, storage_zero<3>>>);

	// does a.dot(b) == a.wedge(b.hodgeDual) ?
	// probably not for non-antisymmetric a and b (since a∧✱b will antisymmetrize a and b)
	// but will it have a 1/k! factor for k-forms a and b?
	// looks like it ...
	// 1-form test
	{
		auto a = float3([](int i) -> float { return i + 1; });
		auto b = float3([](int i) -> float { return 5 - 2*i; });
		ECHO(a);
		ECHO(b);
		ECHO(a.dot(b));
		ECHO(b.dual().expand());
		ECHO(b.dual().dual().expand());
		ECHO(a.wedge(b.dual()).expand());
		ECHO(a.wedge(b.dual()).dual());
		// for 1 forms, a dot b = a wedge star b
		TEST_EQ(a.dot(b), a.wedge(b.dual()).dual());
	}
	// 2-form test
	{
		auto a = float3x3([](int i, int j) -> float { return i + 4 * j; });
		auto b = float3x3([](int i, int j) -> float { return 10 - 3 * i - j; });
		ECHO(a);
		ECHO(b);
		ECHO(a.dot(b));
		ECHO(a.makeSym().expand());
		ECHO(a.makeSym() + a.makeAsym());
		TEST_EQ(a, a.makeSym() + a.makeAsym());
		ECHO(a.makeAsym().expand());
		ECHO(b.makeAsym().expand());
		ECHO(a.makeAsym().dot(b.makeAsym()));
		ECHO(b.dual().expand());
		ECHO(b.dual().dual().expand());
		ECHO(a.wedge(b.dual()).expand());
		ECHO(a.wedge(b.dual()).dual());
		// for 2 forms, asym(a) dot asym(b) = 2 a wedge star b
		TEST_EQ(
			a.makeAsym().dot(b.makeAsym()), 
			2.f * a.wedge(b.dual()).dual()
		);
	}
	// 3-form test
	{
		auto a = floatNaR<3,3>(2.f) + _tensorr<float, 3,3>([](int i, int j, int k) -> float { return 1.f + i + 2. * j - 3. * k; });
		auto b = floatNaR<3,3>(5.f) + _tensorr<float, 3,3>([](int i, int j, int k) -> float { return 4.f * i - 2. * j +  k + 5.f; });
		ECHO(a.makeAsym());
		ECHO(b.makeAsym());
		TEST_EQ(
			a.makeAsym().dot(b.makeAsym()), 
			//6.f * a.wedge(b.dual()).dual()	// b.dual is a float, so a.wedge( b.dual() ) is just mul
			// TODO should I make Tensor::wedge(float) fall back to operator*() ?
			// also should I make Tensor::dual(float) make a _asymR<float,N,N> tensor?  but what dimension would it be?
			6.f * (a * b.dual()).dual()
		);	
	}
}
