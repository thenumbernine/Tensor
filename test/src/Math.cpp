#include "Test/Test.h"


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
	}(inputAType inputAArgs);

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
	}(inputAType inputAArgs, inputBType inputBArgs);

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
}
