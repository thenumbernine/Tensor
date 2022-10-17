#include "Test/Test.h"

// TODO move this to Tensor/Vector.h.h
namespace Tensor {
	using float3x3x3 = _tensorr<float, 3, 3>;
	using float3x3i3 = _tensori<float, storage_vec<3>, storage_ident<3>>;
	using float3i3x3 = _tensori<float, storage_ident<3>, storage_vec<3>>;
	using float3x3s3 = _tensori<float, storage_vec<3>, storage_sym<3>>;
	using float3s3x3 = _tensori<float, storage_sym<3>, storage_vec<3>>;
	using float3x3a3 = _tensori<float, storage_vec<3>, storage_asym<3>>;
	using float3a3x3 = _tensori<float, storage_asym<3>, storage_vec<3>>;
}
namespace TestTotallySymmetric {
	using namespace Tensor;
	using namespace std;

	// _tensorx creation notation
	// test against _tensori so I'm sure it's not just testing itself to itself
	static_assert(is_same_v<_tensorx<float, 3>, _tensori<float, storage_vec<3>>>);																//T_i						float#
	static_assert(is_same_v<_tensorx<float, 3, 3>, _tensori<float, storage_vec<3>, storage_vec<3>>>);											//T_ij						float#x#
	static_assert(is_same_v<_tensorx<float, -'i', 3>, _tensori<float, storage_ident<3>>>);														//δ_ij = δ_(ij)				float#i#
	static_assert(is_same_v<_tensorx<float, -'s', 3>, _tensori<float, storage_sym<3>>>);														//T_ij = T_(ij)				float#s#
	static_assert(is_same_v<_tensorx<float, -'a', 3>, _tensori<float, storage_asym<3>>>);														//T_ij = T_[ij]				float#a#
	static_assert(is_same_v<_tensorx<float, 3, 3, 3>, _tensori<float, storage_vec<3>, storage_vec<3>, storage_vec<3>>>);						//T_ijk						float#x#x#
	static_assert(is_same_v<_tensorx<float, 3, -'i', 3>, _tensori<float, storage_vec<3>, storage_ident<3>>>);									//T_ijk = a_i δ_(jk)
	static_assert(is_same_v<_tensorx<float, 3, -'s', 3>, _tensori<float, storage_vec<3>, storage_sym<3>>>);										//T_ijk = T_i(jk)
	static_assert(is_same_v<_tensorx<float, 3, -'a', 3>, _tensori<float, storage_vec<3>, storage_asym<3>>>);									//T_ijk = T_i[jk]
	static_assert(is_same_v<_tensorx<float, -'i', 3, 3>, _tensori<float, storage_ident<3>, storage_vec<3>>>);									//T_ijk = δ_(ij) b_k
	static_assert(is_same_v<_tensorx<float, -'s', 3, 3>, _tensori<float, storage_sym<3>, storage_vec<3>>>);										//T_ijk = T_(ij)k
	static_assert(is_same_v<_tensorx<float, -'a', 3, 3>, _tensori<float, storage_asym<3>, storage_vec<3>>>);									//T_ijk = T_[ij]_k
	static_assert(is_same_v<_tensorx<float, -'S', 3, 3>, _tensori<float, storage_symR<3, 3>>>);													//T_ijk = T_[ijk]			float#s#s#, # >= 3
	static_assert(is_same_v<_tensorx<float, -'A', 3, 3>, _tensori<float, storage_asymR<3, 3>>>);												//T_ijk = T_[ijk]			float#a#a#, # >= 3
	static_assert(is_same_v<_tensorx<float, 3, 3, 3, 3>, _tensori<float, storage_vec<3>, storage_vec<3>, storage_vec<3>, storage_vec<3>>>);		//T_ijkl
	static_assert(is_same_v<_tensorx<float, 3, 3, -'s', 3>, _tensori<float, storage_vec<3>, storage_vec<3>, storage_sym<3>>>);					//T_ijkl = T_ij(kl)
	static_assert(is_same_v<_tensorx<float, 3, 3, -'i', 3>, _tensori<float, storage_vec<3>, storage_vec<3>, storage_ident<3>>>);				//T_ijkl = a_ij δ_(kl)
	static_assert(is_same_v<_tensorx<float, 3, 3, -'a', 3>, _tensori<float, storage_vec<3>, storage_vec<3>, storage_asym<3>>>);					//T_ijkl = T_ij[kl]
	static_assert(is_same_v<_tensorx<float, 3, -'s', 3, 3>, _tensori<float, storage_vec<3>, storage_sym<3>, storage_vec<3>>>);					//T_ijkl = T_i(jk)l
	static_assert(is_same_v<_tensorx<float, 3, -'i', 3, 3>, _tensori<float, storage_vec<3>, storage_ident<3>, storage_vec<3>>>);				//T_ijkl = a_i δ_(jk) c_l
	static_assert(is_same_v<_tensorx<float, 3, -'a', 3, 3>, _tensori<float, storage_vec<3>, storage_asym<3>, storage_vec<3>>>);					//T_ijkl = T_i[jk]l
	static_assert(is_same_v<_tensorx<float, -'s', 3, 3, 3>, _tensori<float, storage_sym<3>, storage_vec<3>, storage_vec<3>>>);					//T_ijkl = T_(ij)kl
	static_assert(is_same_v<_tensorx<float, -'i', 3, 3, 3>, _tensori<float, storage_ident<3>, storage_vec<3>, storage_vec<3>>>);				//T_ijkl = delta_(ij) b_kl
	static_assert(is_same_v<_tensorx<float, -'s', 3, -'s', 3>, _tensori<float, storage_sym<3>, storage_sym<3>>>);								//T_ijkl = T_(ij)(kl)
	static_assert(is_same_v<_tensorx<float, -'i', 3, -'s', 3>, _tensori<float, storage_ident<3>, storage_sym<3>>>);								//T_ijkl = delta_(ij) b_(kl)
	static_assert(is_same_v<_tensorx<float, -'s', 3, -'a', 3>, _tensori<float, storage_sym<3>, storage_asym<3>>>);								//T_ijkl = T_(ij)[kl]
	static_assert(is_same_v<_tensorx<float, -'i', 3, -'a', 3>, _tensori<float, storage_ident<3>, storage_asym<3>>>);							//T_ijkl = delta_(ij) b_[kl]
	static_assert(is_same_v<_tensorx<float, -'a', 3, 3, 3>, _tensori<float, storage_asym<3>, storage_vec<3>, storage_vec<3>>>);					//T_ijkl = T_[ij]kl
	static_assert(is_same_v<_tensorx<float, -'a', 3, -'s', 3>, _tensori<float, storage_asym<3>, storage_sym<3>>>);								//T_ijkl = T_[ij](kl)
	static_assert(is_same_v<_tensorx<float, -'a', 3, -'i', 3>, _tensori<float, storage_asym<3>, storage_ident<3>>>);							//T_ijkl = a_[ij] delta_(kl)
	static_assert(is_same_v<_tensorx<float, -'a', 3, -'a', 3>, _tensori<float, storage_asym<3>, storage_asym<3>>>);								//T_ijkl = T_[ij][kl]
	static_assert(is_same_v<_tensorx<float, 3, -'S', 3, 3>, _tensori<float, storage_vec<3>, storage_symR<3, 3>>>);								//T_ijkl = T_i(jkl)
	static_assert(is_same_v<_tensorx<float, 3, -'A', 3, 3>, _tensori<float, storage_vec<3>, storage_asymR<3, 3>>>);								//T_ijkl = T_i[jkl]
	static_assert(is_same_v<_tensorx<float, -'S', 3, 3, 3>, _tensori<float, storage_symR<3, 3>, storage_vec<3>>>);								//T_ijkl = T_(ijk)l
	static_assert(is_same_v<_tensorx<float, -'A', 3, 3, 3>, _tensori<float, storage_asymR<3, 3>, storage_vec<3>>>);								//T_ijkl = T_[ijk]l
	static_assert(is_same_v<_tensorx<float, -'S', 3, 4>, _tensori<float, storage_symR<3, 4>>>);													//T_ijkl = T_(ijkl)			float#s#s#s#, # >= 4
	static_assert(is_same_v<_tensorx<float, -'A', 3, 4>, _tensori<float, storage_asymR<3, 4>>>);												//T_ijkl = T_[ijkl]			float#a#a#a#, # >= 4

	// TODO storage helper? if the user chooses storage_symR rank=1 then use storage_vec, for rank=2 use storage_sym ...
	//  however doing _symR<dim,rank>  for rank==1 or rank==2 does static_assert / require fail,
	//  so should I just also let storage_symR fail in the same way?

	//RemoveIndex:
	static_assert(is_same_v<float3a3::RemoveIndex<0>, float3>);
	static_assert(is_same_v<float3a3a3::RemoveIndex<0>, float3a3>);
	// TODO get this to work
	static_assert(is_same_v<float3a3a3::RemoveIndex<1>, float3x3>);
	static_assert(is_same_v<float3a3a3::RemoveIndex<2>, float3a3>);

	
	STATIC_ASSERT_EQ((constexpr_isqrt(0)), 0);
	STATIC_ASSERT_EQ((constexpr_isqrt(1)), 1);
	STATIC_ASSERT_EQ((constexpr_isqrt(2)), 1);
	STATIC_ASSERT_EQ((constexpr_isqrt(3)), 1);
	STATIC_ASSERT_EQ((constexpr_isqrt(4)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(5)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(6)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(7)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(8)), 2);
	STATIC_ASSERT_EQ((constexpr_isqrt(9)), 3);

	static_assert(constexpr_factorial(0) == 1);
	static_assert(constexpr_factorial(1) == 1);
	static_assert(constexpr_factorial(2) == 2);
	static_assert(constexpr_factorial(3) == 6);
	static_assert(constexpr_factorial(4) == 24);

	static_assert(nChooseR(0,0) == 1);
	static_assert(nChooseR(1,0) == 1);
	static_assert(nChooseR(1,1) == 1);
	static_assert(nChooseR(2,0) == 1);
	static_assert(nChooseR(2,1) == 2);
	static_assert(nChooseR(2,2) == 1);
	static_assert(nChooseR(3,0) == 1);
	static_assert(nChooseR(3,1) == 3);
	static_assert(nChooseR(3,2) == 3);
	static_assert(nChooseR(3,3) == 1);
	static_assert(nChooseR(4,0) == 1);
	static_assert(nChooseR(4,1) == 4);
	static_assert(nChooseR(4,2) == 6);
	static_assert(nChooseR(4,3) == 4);
	static_assert(nChooseR(4,4) == 1);
}



void test_TotallySymmetric() {
	/*
	unique indexing of 3s3s3:
000
001 010 100
002 020 200
011 101 110
012 021 102 120 201 210
022 202 220
111
112 121 211
122 212 221
222
	*/
	using float3s3s3 = Tensor::_symR<float, 3, 3>;
	static_assert(sizeof(float3s3s3) == sizeof(float) * 10);
	{
		auto t = float3s3s3();

		t(0,0,0) = 1;
		TEST_EQ(t(0,0,0), 1);

		t(1,0,0) = 2;
		TEST_EQ(t(1,0,0), 2);
		TEST_EQ(t(0,1,0), 2);
		TEST_EQ(t(0,0,1), 2);

		// Iterator needs operator()(intN i)
		auto i = t.begin();
		TEST_EQ(*i, 1);
	}

	{
		auto f = [](int i, int j, int k) -> float { return i + j + k; };
		auto t = float3s3s3(f);
		verifyAccessRank3<decltype(t)>(t, f);
		verifyAccessRank3<decltype(t) const>(t, f);

		// operators need Iterator
		operatorScalarTest(t);
	}

	{
		auto f = [](int i, int j, int k) -> float { return i + j + k; };
		auto t = float3s3s3(f);

		auto v = Tensor::float3{3, 6, 9};
		auto m = v * t;
		static_assert(std::is_same_v<decltype(m), Tensor::float3s3>);
	}

	// make sure call-through works
	{
		using namespace Tensor;
		// t_ijklmn = t_(ij)(klm)n
		auto f = [](int i, int j, int k, int l, int m, int n) -> float {
			return (i+j) - (k+l+m) + n;
		};
		auto t = _tensori<float, storage_sym<3>, storage_symR<3,3>, storage_vec<3>>(f);
		ECHO(t);
		ECHO(t(0,0,0,0,0,0));
	}
}
