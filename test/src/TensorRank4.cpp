#include "Test/Test.h"

void test_TensorRank4() {
	// rank-4


	// vec-vec-vec-vec
	{
		using T = Tensor::tensorr<float, 3, 4>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k+l; };
		auto t = T(f);
		verifyAccessRank4<T>(t, f);
		verifyAccessRank4<T const>(t, f);
	}

	// sym-vec-vec
	{
		using T = Tensor::tensori<float, Tensor::storage_sym<3>, Tensor::storage_vec<3>, Tensor::storage_vec<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k+l; };
		auto t = T(f);
		verifyAccessRank4<T>(t, f);
		verifyAccessRank4<T const>(t, f);
	}

#if 0
	// asym-vec-vec
	{
		using T = Tensor::tensori<float, Tensor::storage_asym<3>, Tensor::storage_vec<3>, Tensor::storage_vec<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i-j+k+l; };
		auto t = T(f);
		verifyAccessRank4<T>(t, f);
		verifyAccessRank4<T const>(t, f);
	}
#endif

	// vec-sym-vec
	{
		using T = Tensor::tensori<float, Tensor::storage_vec<3>, Tensor::storage_sym<3>, Tensor::storage_vec<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k+l; };
		auto t = T(f);
		verifyAccessRank4<T>(t, f);
		verifyAccessRank4<T const>(t, f);
	}

#if 0
	// vec-asym-vec
	{
		using T = Tensor::tensori<float, Tensor::storage_vec<3>, Tensor::storage_asym<3>, Tensor::storage_vec<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j-k+l; };
		auto t = T(f);
		verifyAccessRank4<T>(t, f);
		verifyAccessRank4<T const>(t, f);
	}
#endif

	// vec-vec-sym
	{
		using T = Tensor::tensori<float, Tensor::storage_vec<3>, Tensor::storage_vec<3>, Tensor::storage_sym<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k+l; };
		auto t = T(f);
		verifyAccessRank4<T>(t, f);
		verifyAccessRank4<T const>(t, f);
	}
#if 0
	// vec-vec-asym
	{
		using T = Tensor::tensori<float, Tensor::storage_vec<3>, Tensor::storage_vec<3>, Tensor::storage_asym<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k-l; };
		auto t = T(f);
		verifyAccessRank4<T>(t, f);
		verifyAccessRank4<T const>(t, f);
	}
#endif

	// sym-sym
	{
		using T = Tensor::tensori<float, Tensor::storage_sym<3>, Tensor::storage_sym<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k+l; };
		auto t = T(f);
		verifyAccessRank4<decltype(t)>(t, f);
		verifyAccessRank4<decltype(t) const>(t, f);
	}

#if 0	// asym-asym
	{
		using T = Tensor::tensori<float, Tensor::storage_asym<3>, Tensor::storage_asym<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k+l; };
		auto t = T(f);
		verifyAccessRank4<decltype(t)>(t, f);
		verifyAccessRank4<decltype(t) const>(t, f);
	}
#endif

	{
		using Real = double;
		using Riemann2 = Tensor::tensori<Real, Tensor::storage_asym<2>, Tensor::storage_asym<2>>;
		//using Riemann2 = Tensor::asym<Tensor::asym<Real, 2>, 2>;	// R_[ij][kl]
		//using Riemann2 = Tensor::sym<Tensor::asym<Real, 2>, 2>;	// ... R_(ij)[kl] ...
		// how would I define R_( [ij] [kl ) ... i.e. R_ijkl = R_klij and R_ijkl = -R_jikl ?
		auto r = Riemann2{{1}};
		static_assert(Riemann2::rank == 4);
		static_assert(Riemann2::dim<0> == 2);
		static_assert(Riemann2::dim<1> == 2);
		static_assert(Riemann2::dim<2> == 2);
		static_assert(Riemann2::dim<3> == 2);
		static_assert(Riemann2::numNestings == 2);
		static_assert(Riemann2::count<0> == 1);
		static_assert(Riemann2::count<1> == 1);
		static_assert(sizeof(Riemann2) == sizeof(Real));
		auto r00 = r(0,0);	// this type will be a ZERO AntiSymRef wrapper around ... nothing ...
		ECHO(r00);
		TEST_EQ(r00, (Tensor::AntiSymRef<Tensor::asym<Real, 2>>()));	// r(0,0) is this type
		TEST_EQ(r00, (Tensor::asym<Real, 2>{}));	// ... and r(0,0)'s operator== accepts its wrapped type
		TEST_EQ(r00(0,0), (Tensor::AntiSymRef<Real>()));	// r(0,0)(0,0) is this
		TEST_EQ(r00(0,0).how, Tensor::Sign::ZERO);
		TEST_EQ(r00(0,0), 0.);
		TEST_EQ(r00(0,1), 0.);
		TEST_EQ(r00(1,0), 0.);
		TEST_EQ(r00(1,1), 0.);
		auto r01 = r(0,1);	// this will point to the positive r.x_y element
		TEST_EQ(r01, (Tensor::asym<Real, 2>{1}));
		TEST_EQ(r01(0,0), 0);	//why would this get a bad ref?
		TEST_EQ(r01(0,1), 1);
		TEST_EQ(r01(1,0), -1);
		TEST_EQ(r01(1,1), 0);
		auto r10 = r(1,0);
		TEST_EQ(r10, (Tensor::asym<Real, 2>{-1}));
		TEST_EQ(r10(0,0), 0);
		TEST_EQ(r10(0,1), -1);
		TEST_EQ(r10(1,0), 1);
		TEST_EQ(r10(1,1), 0);
		auto r11 = r(1,1);
		TEST_EQ(r11(0,0), 0.);
		TEST_EQ(r11(0,1), 0.);
		TEST_EQ(r11(1,0), 0.);
		TEST_EQ(r11(1,1), 0.);
	}
	{
		constexpr int N = 3;
		using Riemann3 = Tensor::tensori<double, Tensor::storage_asym<N>, Tensor::storage_asym<N>>;
		auto r = Riemann3();
		static_assert(Riemann3::rank == 4);
		static_assert(Riemann3::dim<0> == N);
		static_assert(Riemann3::dim<1> == N);
		static_assert(Riemann3::dim<2> == N);
		static_assert(Riemann3::dim<3> == N);
		static_assert(Riemann3::numNestings == 2);
		static_assert(Riemann3::count<0> == 3);	//3x3 antisymmetric has 3 unique components
		static_assert(Riemann3::count<1> == 3);
		//TODO some future work: R_ijkl = R_klij, so it's also symmetri between 1&2 and 3&4 ...
		// ... and optimizing for those should put us at only 6 unique values instead of 9
		static_assert(sizeof(Riemann3) == sizeof(double) * 9);

		double e = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < i; ++j) {
				for (int k = 0; k < N; ++k) {
					for (int l = 0; l < N; ++l) {
						r(i,j)(k,l) = ++e;
						if (i == j || k == l) {
							TEST_EQ(r(i,j)(k,l), 0);
						} else {
							TEST_EQ(r(i,j)(k,l), e);
							TEST_EQ(r(i,j)(l,k), -e);
							TEST_EQ(r(j,i)(k,l), -e);
							TEST_EQ(r(j,i)(l,k), e);

							TEST_EQ(r(i,j,k,l), e);
							TEST_EQ(r(i,j,l,k), -e);
							TEST_EQ(r(j,i,k,l), -e);
							TEST_EQ(r(j,i,l,k), e);
						}
					}
				}
			}
		}

// TODO change tensor generic ctor to (requires tensors  and) accept any type, iterate over write elements, assign one by one.

//		auto m = Tensor::ExpandAllIndexes<Riemann3>([&](Tensor::int4 i) -> double {
//			return r(i);
//		});
	}

	{
		// TODO verify 3- nestings deep of antisym works
	}

	// old libraries' tests
	{
		using Real = double;
		using Vector = Tensor::tensor<Real,3>;

		Vector v = {1,2,3};
		TEST_EQ(v, Tensor::double3(1,2,3));

		using Metric = Tensor::tensori<Real,Tensor::storage_sym<3>>;
		Metric g;
		for (int i = 0; i < 3; ++i) {
			g(i,i) = 1;
		}
		TEST_EQ(g, Metric(1,0,1,0,0,1));

		using Matrix = Tensor::tensor<Real,3,3>;
		Matrix h;
		int index = 0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				h(i,j) = ++index;
			}
		}
		TEST_EQ(h, (Matrix{{1,2,3},{4,5,6},{7,8,9}}));

		//iterator access
		int j = 0;
		Tensor::tensor<Real, 3,3,3> ta;
		for (auto i = ta.begin(); i != ta.end(); ++i) {
			*i = j++;
		}
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				for (int k = 0; k < 3; ++k) {
					if constexpr (Tensor::int2::useReadIteratorOuter) {
						TEST_EQ(ta(i,j,k), k + 3 * (j + 3 * i));
					} else {
						TEST_EQ(ta(i,j,k), i + 3 * (j + 3 * k));
					}
				}
			}
		}

		//subtensor access not working
		Tensor::tensor<Real,3,3> tb;
		for (auto i = tb.begin(); i != tb.end(); ++i) *i = 2.f;
		TEST_EQ(tb, Matrix(2.f));
		ta(0) = tb;
		TEST_EQ(ta, (Tensor::tensor<Real,3,3,3>{
			{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}},
			ta(1),//{{1, 10, 19}, {4, 13, 22}, {7, 16, 25}}, // these are whatever the original ta was
			ta(2),//{{2, 11, 20}, {5, 14, 23}, {8, 17, 26}}
		} ));
		Tensor::tensor<Real, 3> tc;
		for (auto i = tc.begin(); i != tc.end(); ++i) *i = 3.;
		TEST_EQ(Tensor::double3(3), tc);
		ta(0,0) = tc;
		TEST_EQ(ta, (Tensor::tensor<Real,3,3,3>{
			{{3, 3, 3}, {2, 2, 2}, {2, 2, 2}},
			ta(1),//{{1, 10, 19}, {4, 13, 22}, {7, 16, 25}},
			ta(2),//{{2, 11, 20}, {5, 14, 23}, {8, 17, 26}}
		}));

		//inverse
		Matrix m;
		for (int i = 0; i < m.dim<0>; ++i) {
			for (int j = 0; j < m.dim<1>; ++j) {
				m(i,j) = i == j ? 1 : 0;
			}
		}

		// convert the sym diagonal to mat
		// TODO operator== between matrices
		auto d = diagonal(Tensor::vec<Real, m.dim<0>>(1));
		TEST_EQ(m, d);
		TEST_EQ(m, (Matrix{{1,0,0},{0,1,0},{0,0,1}}));
		TEST_EQ(Tensor::determinant(m), 1);
	}

	{
		using namespace Tensor;
		using I = ident<float, 1>;
		{
			auto a = ident<float, 1>(4);
			TEST_EQ(a, I(4));
		}
		{
			auto a = ident<float, 1>();
			TEST_EQ(a, I(0));
		}
		{
			auto t = ident<float, 1>{5};
			t.s[0] = 1;
			TEST_EQ(t, I(1));
			t[0][0] = 2;
			TEST_EQ(t, I(2));
			t(0,0) = 3;
			TEST_EQ(t, I(3));
			t(int2(0,0)) = 4;
			TEST_EQ(t, I(4));
		}
#if 0 // TODO vector * ident3x3
		{
			auto x = float3{1,2,3} * ident<float,3>(1.);
			ECHO(x);
		}
#endif
	}

	// can you vector non-numeric types?
	{
		using namespace Tensor;
		using namespace std;
		// "attempt to use a deleted function"
		// "destructor of 'vec<..., 3>' is implicitly deleted because variant field '' has a non-trivial destructor"
		//auto t = vec<function<void()>, 3>(); // fails
		//auto t = vec<optional<function<void()>>, 3>(); // fails
		auto t = vec<optional<function<void()>>, 5>(); // works
		//auto t = vec<std::array<int, 3>, 4>(); // works, but meh
		//ECHO(t);
		//auto t = vec<std::string, 4>(); // hmm  my fixed-size specialization vectors can't ctor
		//auto t = vec<std::string, 5>(); // but the default case works fine
		//ECHO(t);
	}

	// multiply (operator*) and tensor functions

	{
		Tensor::float3x3 m = {
			{1,2,3},
			{4,5,6},
			{7,8,9},
		};

		// TODO since operator* is based on tensor products, maybe put it later?
		//  put all the tensor operations at the end
		// operator *
		{
			auto a = Tensor::int2{7, -2};
			TEST_EQ((Tensor::contract<0,0>(a)), 5);
			auto b = Tensor::int2x2
				{{6, 9},
				{6, -6}};
			TEST_EQ((Tensor::contract<0,1>(b)), 0);
			TEST_EQ((Tensor::contract<1,0>(b)), 0);
			static_assert(std::is_same_v<Tensor::int2x2::RemoveIndex<0>, Tensor::int2>);
			ECHO((Tensor::contract<0,0>(b)));
			ECHO((Tensor::contract<1,1>(b)));

			auto aouterb = Tensor::tensorr<int,2,3>{{{42, 63}, {42, -42}}, {{-12, -18}, {-12, 12}}};
			TEST_EQ(outer(a,b), aouterb);
			ECHO((Tensor::contract<0,0>(aouterb)));

			static_assert(std::is_same_v<Tensor::tensorr<int,2,3>::RemoveIndex<0>, Tensor::int2x2>);
			static_assert(std::is_same_v<Tensor::tensorr<int,2,3>::RemoveIndex<1>, Tensor::int2x2>);
			static_assert(std::is_same_v<Tensor::tensorr<int,2,3>::RemoveIndex<2>, Tensor::int2x2>);
			static_assert(std::is_same_v<Tensor::int2x2::RemoveIndex<0>, Tensor::int2>);
			static_assert(std::is_same_v<Tensor::int2x2::RemoveIndex<1>, Tensor::int2>);
			static_assert(std::is_same_v<Tensor::tensorr<int,2,3>::RemoveIndex<0,1>, Tensor::int2>);
			ECHO((Tensor::contract<0,1>(aouterb)));

			ECHO((Tensor::contract<0,2>(aouterb)));
			ECHO((Tensor::contract<1,0>(aouterb)));
			ECHO((Tensor::contract<1,1>(aouterb)));
			ECHO((Tensor::contract<1,2>(aouterb)));
			ECHO((Tensor::contract<2,0>(aouterb)));
			ECHO((Tensor::contract<2,1>(aouterb)));
			ECHO((Tensor::contract<2,2>(aouterb)));
			auto atimesb = Tensor::int2{30, 75};
			TEST_EQ(a * b, atimesb);

			TEST_EQ( (Tensor::int2{-3, 6}
				* Tensor::int2x3{
					{-5, 0, -3},
					{1, 3, 0}}),
				(Tensor::int3{21, 18, 9}))
			TEST_EQ( (Tensor::int2{9, 9}
				* Tensor::int2x4{
					{3, -8, -10, -8},
					{5, 2, -5, 6}}),
				(Tensor::int4{72, -54, -135, -18}))
			TEST_EQ( (Tensor::int3{-7, -2, -8}
				* Tensor::int3x2{
					{-8, 0},
					{10, 7},
					{-6, 2}}),
				(Tensor::int2{84, -30}))
			TEST_EQ( (Tensor::int3{-4, 3, 1}
				* Tensor::int3x3{
					{0, 6, -2},
					{10, 1, 8},
					{-4, 6, -5}}),
				(Tensor::int3{26, -15, 27}))
			TEST_EQ( (Tensor::int3{-3, 6, 9}
				* Tensor::int3x4{
					{-9, -9, 8, -10},
					{9, -6, -3, -1},
					{1, 3, -9, -9}}),
				(Tensor::int4{90, 18, -123, -57}))
			TEST_EQ( (Tensor::int4{-5, 10, 8, 7}
				* Tensor::int4x2{
					{-5, 0},
					{1, 4},
					{-3, 1},
					{5, -10}}),
				(Tensor::int2{46, -22}))
			TEST_EQ( (Tensor::int4{-1, 9, 9, 5}
				* Tensor::int4x3{
					{-5, 4, 7},
					{5, -7, -4},
					{3, -1, -6},
					{-3, 8, 8}}),
				(Tensor::int3{62, -36, -57}))
			TEST_EQ( (Tensor::int4{-3, 4, 10, 2}
				* Tensor::int4x4{
					{3, -2, 0, -7},
					{8, 7, -6, 8},
					{-1, 4, 9, 3},
					{9, 9, 9, -1}}),
				(Tensor::int4{31, 92, 84, 81}))
		}

		// TODO make sure operator* matrix/vector, matrix/matrix, vector/matrix works
		// TODO I don't think I have marix *= working yet

		auto m2 = elemMul(m,m);
		for (int i = 0; i < m.dim<0>; ++i) {
			for (int j = 0; j < m.dim<1>; ++j) {
				TEST_EQ(m2(i,j), m(i,j) * m(i,j));
			}
		}

		//determinant

		TEST_EQ(determinant(m), 0);

		// transpose

		TEST_EQ(Tensor::float3x3(
			{1,2,3},
			{4,5,6},
			{7,8,9}
		), Tensor::transpose(Tensor::float3x3(
			{1,4,7},
			{2,5,8},
			{3,6,9}
		)));

		TEST_EQ(Tensor::trace(Tensor::float3x3(
			{1,2,3},
			{4,5,6},
			{7,8,9}
		)), 15);
	}

	// TODO this all goes in a tensor-math test case
	{
		using float3x3x3x3 = Tensor::tensorr<float, 3, 4>;
		auto a = float3x3x3x3([](int i, int j, int k, int l) -> float {
			return i - 2 * j + 3 * k - 4 * l;
		});
		TEST_EQ(a, (float3x3x3x3
			{{{{0, -4, -8},
			{3, -1, -5},
			{6, 2, -2}},
			{{-2, -6, -10},
			{1, -3, -7},
			{4, 0, -4}},
			{{-4, -8, -12},
			{-1, -5, -9},
			{2, -2, -6}}},
			{{{1, -3, -7},
			{4, 0, -4},
			{7, 3, -1}},
			{{-1, -5, -9},
			{2, -2, -6},
			{5, 1, -3}},
			{{-3, -7, -11},
			{0, -4, -8},
			{3, -1, -5}}},
			{{{2, -2, -6},
			{5, 1, -3},
			{8, 4, 0}},
			{{0, -4, -8},
			{3, -1, -5},
			{6, 2, -2}},
			{{-2, -6, -10},
			{1, -3, -7},
			{4, 0, -4}}}}
		));
		auto b = float3x3x3x3([](int i, int j, int k, int l) -> float {
			return 5 * l - 6 * k + 7 * j - 8 * i;
		});
		TEST_EQ(b, (float3x3x3x3
			{{{{0, 5, 10},
			{-6, -1, 4},
			{-12, -7, -2}},
			{{7, 12, 17},
			{1, 6, 11},
			{-5, 0, 5}},
			{{14, 19, 24},
			{8, 13, 18},
			{2, 7, 12}}},
			{{{-8, -3, 2},
			{-14, -9, -4},
			{-20, -15, -10}},
			{{-1, 4, 9},
			{-7, -2, 3},
			{-13, -8, -3}},
			{{6, 11, 16},
			{0, 5, 10},
			{-6, -1, 4}}},
			{{{-16, -11, -6},
			{-22, -17, -12},
			{-28, -23, -18}},
			{{-9, -4, 1},
			{-15, -10, -5},
			{-21, -16, -11}},
			{{-2, 3, 8},
			{-8, -3, 2},
			{-14, -9, -4}}}}
		));

		//c_ijkl = a_ijmn b_mnkl
		auto c = Tensor::interior<2>(a, b);
		TEST_EQ(c, (float3x3x3x3
			{{{{-303, -348, -393},
			{-249, -294, -339},
			{-195, -240, -285}},
			{{-285, -420, -555},
			{-123, -258, -393},
			{39, -96, -231}},
			{{-267, -492, -717},
			{3, -222, -447},
			{273, 48, -177}}},
			{{{-312, -312, -312},
			{-312, -312, -312},
			{-312, -312, -312}},
			{{-294, -384, -474},
			{-186, -276, -366},
			{-78, -168, -258}},
			{{-276, -456, -636},
			{-60, -240, -420},
			{156, -24, -204}}},
			{{{-321, -276, -231},
			{-375, -330, -285},
			{-429, -384, -339}},
			{{-303, -348, -393},
			{-249, -294, -339},
			{-195, -240, -285}},
			{{-285, -420, -555},
			{-123, -258, -393},
			{39, -96, -231}}}}
		));
	}

	{
		using real = double;

		// turns out rank 1x1x...xN ctors doesn't' work ... unless it's 1x1x ... x1
	#if 0
		// TODO why isn't this working?
		using real2 = Tensor::tensor<real, 2>;
		auto i = real2{1,3};
		auto j = real2{2,4};
		using real1x2 = Tensor::tensor<real, 1, 2>;
	#if 1	// works
		auto ii = real1x2{i};
		auto jj = real1x2{j};
	#endif
	#if 0	// fails
		auto ii = real1x2(i);	// doesn't work
		auto jj = real1x2(j);
	#endif
		TEST_EQ(Tensor::inner(ii, jj), 14);
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 1, 3>{{1,3,2}}, Tensor::tensor<real, 1, 3>{{2,4,3}}), 20);
	#endif



		// rank-1 vectors
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 1>{3}, Tensor::tensor<real, 1>{4}), 12);
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 2>{1,3}, Tensor::tensor<real, 2>{2,4}), 14);
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 3>{1,3,2}, Tensor::tensor<real, 3>{2,4,3}), 20);

		// rank-2 dense vectors
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 1, 1>{{3}}, Tensor::tensor<real, 1, 1>{{5}}), 15);
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 2, 1>{{1},{3}}, Tensor::tensor<real, 2, 1>{{2},{4}}), 14);

		TEST_EQ(Tensor::inner(Tensor::tensor<real, 2, 2>{{1, 2},{3, 4}}, Tensor::tensor<real, 2, 2>{{4, 3},{2, 1}}), 20);

		TEST_EQ(Tensor::inner(Tensor::tensor<real, 3, 1>{{1},{3},{2}}, Tensor::tensor<real, 3, 1>{{2},{4},{3}}), 20);

		TEST_EQ(Tensor::inner(Tensor::tensor<real, 3, 2>{{1,2},{3,4},{2,5}}, Tensor::tensor<real, 3, 2>{{2,1},{4,2},{3,3}}), 45);
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 3, 3>{{1,2,3},{4,5,6},{7,8,9}}, Tensor::tensor<real, 3, 3>{{9,8,7},{6,5,4},{3,2,1}}), 165);


		// rank-3 dense

		// rank-2 ident

		TEST_EQ(Tensor::inner(Tensor::ident<real, 3>(2), Tensor::ident<real, 3>(3)), 18);
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 3, 3>{{2,0,0},{0,2,0},{0,0,2}}, Tensor::ident<real, 3>(3)), 18);
		TEST_EQ(Tensor::inner(Tensor::tensor<real, 3, 3>{{2,0,0},{0,2,0},{0,0,2}}, Tensor::tensor<real, 3, 3>{{3,0,0},{0,3,0},{0,0,3}}), 18);

		TEST_EQ(Tensor::inner(Tensor::tensorx<real, -'z', 3, 3>(), Tensor::ident<real, 3>(3)), 0);

		// rank-2 sym*sym
		// rank-2 sym*asym
		TEST_EQ(Tensor::inner(Tensor::sym<real,3>(2), Tensor::asym<real, 3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::asym<real,3>(2), Tensor::sym<real, 3>(3)), 0);
		// rank-2 asym*sym
		// rank-2 asym*asym

		// rank-3 same
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'s',3,3>(2), Tensor::tensorx<real,-'a',3,3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'a',3,3>(2), Tensor::tensorx<real,-'s',3,3>(3)), 0);

		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'S',3,3>(2), Tensor::tensorx<real,-'a',3,3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'S',3,3>(2), Tensor::tensorx<real,-'A',3,3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'A',3,3>(2), Tensor::tensorx<real,-'s',3,3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'A',3,3>(2), Tensor::tensorx<real,-'S',3,3>(3)), 0);

		TEST_EQ(Tensor::inner(Tensor::tensorx<real,3,-'s',3>(2), Tensor::tensorx<real,3,-'a',3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,3,-'a',3>(2), Tensor::tensorx<real,3,-'s',3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'S',3,3>(2), Tensor::tensorx<real,3,-'a',3>(3)), 0);
		TEST_EQ(Tensor::inner(Tensor::tensorx<real,-'A',3,3>(2), Tensor::tensorx<real,3,-'s',3>(3)), 0);

		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'s',3,3>, Tensor::tensorx<real,-'a',3,3>>));
		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'a',3,3>, Tensor::tensorx<real,-'s',3,3>>));

		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'S',3,3>, Tensor::tensorx<real,-'a',3,3>>));
		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'S',3,3>, Tensor::tensorx<real,-'A',3,3>>));
		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'A',3,3>, Tensor::tensorx<real,-'s',3,3>>));
		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'A',3,3>, Tensor::tensorx<real,-'S',3,3>>));

		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,3,-'s',3>, Tensor::tensorx<real,3,-'a',3>>));
		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,3,-'a',3>, Tensor::tensorx<real,3,-'s',3>>));
		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'S',3,3>, Tensor::tensorx<real,3,-'a',3>>));
		TEST_BOOL((Tensor::hasMatchingSymAndAsymIndexes<Tensor::tensorx<real,-'A',3,3>, Tensor::tensorx<real,3,-'s',3>>));

		// rank-4 same

		// rank-4 ident outer rank-2


	}
}
