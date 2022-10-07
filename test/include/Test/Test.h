#pragma once

#include "Tensor/Tensor.h"
#include "Tensor/Inverse.h"
#include "Common/Test.h"

template<typename T>
T sign (T x) {
	return x == T{} ? T{} : (x < T{} ? (T)-1 : (T)1);
};

template<typename T>
void operatorScalarTest(T const & t) {
	using S = typename T::Scalar;
	TEST_EQ(t + (S)0, t);
	TEST_EQ((S)1 + t, t + T((S)1));
	TEST_EQ(t + T(), t);
	TEST_EQ(t - T(), t);
	TEST_EQ((S)0 - t, -t);
	TEST_EQ(t - (S)0, t);
	TEST_EQ(t - (S)1, t - T((S)1));
	TEST_EQ(t - t, T());
	TEST_EQ(t - t * (S)2, -t);
	TEST_EQ(t - (S)2 * t, -t);
	TEST_EQ(t * (S)1, t);
	TEST_EQ(t * (S)-1, -t);
	TEST_EQ((S)-1 * t, -t);
	TEST_EQ(t * (S)2, t + t);
	TEST_EQ(t * (S)0, T());
	TEST_EQ(t / (S)1, t);
	TEST_EQ(t / (S).5, (S)2 * t);
	//TEST_EQ(t / t, T((S)1)); // if t(I) == 0 then this gives nan ... so ...
	TEST_EQ(t / T((S)1), t);
}

template<typename T>
void operatorMatrixTest() {
	static_assert(T::rank >= 2);
}
