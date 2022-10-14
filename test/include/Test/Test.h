#pragma once

#include "Tensor/Tensor.h"
#include "Common/Test.h"

void test_Index();
void test_AntiSymRef();
void test_Vector();
void test_Quat();
void test_Identity();
void test_Matrix();
void test_Symmetric();
void test_Antisymmetric();
void test_TensorRank3();
void test_TensorRank4();
void test_TotallySymmetric();
void test_TotallyAntisymmetric();
void test_Math();

template<typename T>
T sign (T x) {
	return x == T{} ? T{} : (x < T{} ? (T)-1 : (T)1);
};

template<typename T>
void operatorScalarTest(T const & t) {
	using S = typename T::Scalar;
	constexpr bool sumTypeMatches = std::is_same_v<
		T,
		typename T::SumWithScalarResult
	>;
	TEST_EQ(t + (S)0, t);
	// if the sum types don't match then constructoring a T from the scalar 1 might not give us the same as t + 1
	if constexpr (sumTypeMatches) {
		TEST_EQ((S)1 + t, t + T((S)1));
	}
	TEST_EQ(t + T(), t);
	TEST_EQ(t - T(), t);
	TEST_EQ((S)0 - t, -t);
	TEST_EQ(t - (S)0, t);
	if constexpr (sumTypeMatches) {
		TEST_EQ(t - (S)1, t - T((S)1));
	}
	TEST_EQ(t - t, T());
	TEST_EQ(t - t * (S)2, -t);
	TEST_EQ(t - (S)2 * t, -t);
	TEST_EQ(t * (S)1, t);
	TEST_EQ(t * (S)-1, -t);
	TEST_EQ((S)-1 * t, -t);
	TEST_EQ(t * (S)2, t + t);
	TEST_EQ(t * (S)0, T());
	TEST_EQ(t / (S)1, t);
	// hmm, why is this failing
	TEST_EQ(t / (S).5, (S)2 * t);
	//TEST_EQ(t / t, T((S)1)); // if t(I) == 0 then this gives nan ... so ...
	TEST_EQ(t / T((S)1), t);
}

template<typename T>
void operatorMatrixTest() {
	static_assert(T::rank >= 2);
}

template<typename T, typename F>
void verifyAccessRank1(T & t, F f) {
	for (int i = 0; i < T::template dim<0>; ++i) {
		typename T::Scalar x = f(i);
		// various [] and (int...) and (intN)
		TEST_EQ(t(i), x);
		TEST_EQ(t(Tensor::intN<1>(i)), x);
		TEST_EQ(t[i], x);
		TEST_EQ(t.s[i], x);
		if constexpr (!std::is_const_v<T>) {
			t(i) = x;
			t(Tensor::intN<1>(i)) = x;
			t[i] = x;
			t.s[i] = x;
		}
	}
}

template<typename T, typename F>
void verifyAccessRank2(T & t, F f) {
	for (int i = 0; i < T::template dim<0>; ++i) {
		for (int j = 0; j < T::template dim<1>; ++j) {
			typename T::Scalar x = f(i,j);
			TEST_EQ(t(i)(j), x);
			TEST_EQ(t(i,j), x);
			TEST_EQ(t(Tensor::int2(i,j)), x);
			TEST_EQ(t[i](j), x);
			TEST_EQ(t(i)[j], x);
			TEST_EQ(t[i][j], x);
			if constexpr (!std::is_const_v<T>) {
				t(i)(j) = x;
				t(i,j) = x;
				t(Tensor::int2(i,j)) = x;
				t[i](j) = x;
				t(i)[j] = x;
				t[i][j] = x;
			}
		}
	}
}

template<typename T, typename F>
void verifyAccessRank3(T & t, F f) {
	for (int i = 0; i < T::template dim<0>; ++i) {
		for (int j = 0; j < T::template dim<1>; ++j) {
			for (int k = 0; k < T::template dim<2>; ++k) {
				float x = f(i,j,k);
			
				// for _vec interchangeability , do by grouping first then () then [] instead of by () then [] then grouping
				//()()() and any possible merged ()'s
				TEST_EQ(t(i)(j)(k), x);
				TEST_EQ(t[i](j)(k), x);
				TEST_EQ(t(i)[j](k), x);
				TEST_EQ(t(i)(j)[k], x);
				TEST_EQ(t[i][j](k), x);
				TEST_EQ(t[i](j)[k], x);
				TEST_EQ(t(i)[j][k], x);
				TEST_EQ(t[i][j][k], x);
				TEST_EQ(t(i,j)(k), x);
				TEST_EQ(t(i,j)[k], x);
				TEST_EQ(t(Tensor::int2(i,j))(k), x);
				TEST_EQ(t(Tensor::int2(i,j))[k], x);
				TEST_EQ(t(i)(j,k), x);
				TEST_EQ(t[i](j,k), x);
				TEST_EQ(t(i)(Tensor::int2(j,k)), x);
				TEST_EQ(t[i](Tensor::int2(j,k)), x);
				TEST_EQ(t(i,j,k), x);
				TEST_EQ(t(Tensor::int3(i,j,k)), x);
			
				if constexpr (!std::is_const_v<T>) {
					t(i)(j)(k) = x;
					t[i](j)(k) = x;
					t(i)[j](k) = x;
					t(i)(j)[k] = x;
					t[i][j](k) = x;
					t[i](j)[k] = x;
					t(i)[j][k] = x;
					t[i][j][k] = x;
					t(i,j)(k) = x;
					t(i,j)[k] = x;
					t(Tensor::int2(i,j))(k) = x;
					t(Tensor::int2(i,j))[k] = x;
					t(i)(j,k) = x;
					t[i](j,k) = x;
					t(i)(Tensor::int2(j,k)) = x;
					t[i](Tensor::int2(j,k)) = x;
					t(i,j,k) = x;
					t(Tensor::int3(i,j,k)) = x;
				}
			}
		}
	}
}

template<typename T, typename F>
void verifyAccessRank4(T & t, F f){
	for (int i = 0; i < T::template dim<0>; ++i) {
		for (int j = 0; j < T::template dim<1>; ++j) {
			for (int k = 0; k < T::template dim<2>; ++k) {
				for (int l = 0; l < T::template dim<3>; ++l) {
					float x = f(i,j,k,l);
					TEST_EQ(t(i)(j)(k)(l), x);
					TEST_EQ(t[i](j)(k)(l), x);
					TEST_EQ(t(i)[j](k)(l), x);
					TEST_EQ(t(i)(j)[k](l), x);
					TEST_EQ(t(i)(j)(k)[l], x);
					TEST_EQ(t[i][j](k)(l), x);
					TEST_EQ(t[i](j)[k](l), x);
					TEST_EQ(t[i](j)(k)[l], x);
					TEST_EQ(t(i)[j][k](l), x);
					TEST_EQ(t(i)[j](k)[l], x);
					TEST_EQ(t(i)(j)[k][l], x);
					TEST_EQ(t[i][j][k](l), x);
					TEST_EQ(t[i][j](k)[l], x);
					TEST_EQ(t[i](j)[k][l], x);
					TEST_EQ(t(i)[j][k][l], x);
					TEST_EQ(t[i][j][k][l], x);
					
					TEST_EQ(t(i,j)(k)(l), x);
					TEST_EQ(t(i,j)[k](l), x);
					TEST_EQ(t(i,j)(k)[l], x);
					TEST_EQ(t(i,j)(k)[l], x);
					TEST_EQ(t(i,j)[k][l], x);
					TEST_EQ(t(Tensor::int2(i,j))(k)(l), x);
					TEST_EQ(t(Tensor::int2(i,j))[k](l), x);
					TEST_EQ(t(Tensor::int2(i,j))(k)[l], x);
					TEST_EQ(t(Tensor::int2(i,j))(k)[l], x);
					TEST_EQ(t(Tensor::int2(i,j))[k][l], x);

					TEST_EQ(t(i,j)(k,l), x);
					TEST_EQ(t(Tensor::int2(i,j))(k,l), x);
					TEST_EQ(t(i,j)(Tensor::int2(k,l)), x);
					TEST_EQ(t(Tensor::int2(i,j))(Tensor::int2(k,l)), x);
					
					TEST_EQ(t(i)(j,k)(l), x);
					TEST_EQ(t[i](j,k)(l), x);
					TEST_EQ(t(i)(j,k)[l], x);
					TEST_EQ(t[i](j,k)[l], x);
					TEST_EQ(t(i)(Tensor::int2(j,k))(l), x);
					TEST_EQ(t[i](Tensor::int2(j,k))(l), x);
					TEST_EQ(t(i)(Tensor::int2(j,k))[l], x);
					TEST_EQ(t[i](Tensor::int2(j,k))[l], x);

					TEST_EQ(t(i)(j)(k,l), x);
					TEST_EQ(t[i](j)(k,l), x);
					TEST_EQ(t(i)[j](k,l), x);
					TEST_EQ(t[i][j](k,l), x);
					TEST_EQ(t(i)(j)(Tensor::int2(k,l)), x);
					TEST_EQ(t[i](j)(Tensor::int2(k,l)), x);
					TEST_EQ(t(i)[j](Tensor::int2(k,l)), x);
					TEST_EQ(t[i][j](Tensor::int2(k,l)), x);	

					TEST_EQ(t(i)(j,k,l), x);
					TEST_EQ(t[i](j,k,l), x);
					TEST_EQ(t(i)(Tensor::int3(j,k,l)), x);
					TEST_EQ(t[i](Tensor::int3(j,k,l)), x);
					
					TEST_EQ(t(i,j,k)(l), x);
					TEST_EQ(t(Tensor::int3(i,j,k))(l), x);
					
					TEST_EQ(t(i,j,k,l), x);
					TEST_EQ(t(Tensor::int4(i,j,k,l)), x);
				}
			}
		}
	}
}
