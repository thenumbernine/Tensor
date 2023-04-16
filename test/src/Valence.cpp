#include "Test/Test.h"
#include "Tensor/Valence.h"

// static assert failure ... from creating the template argument
// if there was a static_assert failure within the body of the type_identity dependent on the template parameter ... that's deferred ... 

void test_Valence() {
	using real = double;
	using namespace Tensor;

	/*
	TODO valence indicators ...
	right now I have 'u' vs 'd'
	but it'd be nice to specify
	*/

	// a^i_j
	/* // constructor with valence as child class of tensor:
	auto a = valence<
		tensor<real, 3, 3>,
		'u', 'd'
	>{{1,2,3},{4,5,6},{7,8,9}};
	*/
	// constructor for valence as wrapper class of tensor:
	auto a = valence<'u', 'd'>(
		tensor<real, 3, 3>{{1,2,3},{4,5,6},{7,8,9}}
	);
	ECHO(a);

	// g_ij
	auto g = valence<'d', 'd'>(
		tensor<real, 3, 3>{{-1,0,0},{0,1,0},{0,0,1}}
	);
	ECHO(g);

	// g^ij
	// if I make the valence wrapper a subclass of the tensor then I still have to explicit-cast this
	//auto tmp = inverse((tensor<real,3,3>)g);
	// so maybe I should just use a member, and use -> and * with it?
	auto tmp = inverse(*g);
	auto gU = valence<'u', 'u'>(
		tensor<real, 3, 3>(
			[&](int i, int j) -> real { return tmp(i,j); }
		)
	);
	ECHO(gU);

#if 0
	// static_assert failure of a * g;
	auto aerr = a * g;
#endif

	// a_ij
	auto aL = g * a;
	auto aLcheck = (valence<'d', 'd'>(tensor<real, 3, 3>{{-1,-2,-3},{4,5,6},{7,8,9}}));
	TEST_EQ(aL, aLcheck);

#if 0
	// static_assert failure of gU * a
	auto aerr = gU * a;
#endif

	// a^ij
	auto aU = a * gU;
	TEST_EQ(aU, (valence<'u', 'u'>(tensor<real, 3, 3>{{-1,2,3},{-4,5,6},{-7,8,9}})));

#if 0
	//static assert failure - valence mismatch
	auto bL = aL + aU;
	ECHO(bL)
#endif
}
