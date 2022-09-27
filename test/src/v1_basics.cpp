#include "Tensor/v1/Tensor.h"
//#include "Tensor/v1/Inverse.h"
#include "Common/Test.h"
#include <iostream>

namespace Tensor {
using namespace Tensor::v1;
}

void test_v1_vectors() {
	using Real = double;
	using Vector = Tensor::Vector<Real, 3>;

	//arg ctor works
	Vector a(1,2,3);
	
	//bracket ctor works
	Vector b = {4,5,6};

	//access
	TEST_EQ(a(0), 1);
	TEST_EQ(a[0], 1);

	//make sure GenericArray functionality works
	TEST_EQ(Vector(1), Vector(1,1,1));
	TEST_EQ(Vector(1,2), Vector(1,2,0));
	TEST_EQ(b + a, Vector(5,7,9));
	TEST_EQ(b - a, Vector(3,3,3));
	TEST_EQ(b * a, Vector(4, 10, 18));
	TEST_EQ(Vector(2,4,6)/Vector(1,2,3), Vector(2,2,2));
	TEST_EQ(b * 2., Vector(8, 10, 12));
	TEST_EQ(Vector(2,4,6)/2., Vector(1,2,3));
}

template<typename InputType>
typename InputType::Type determinant33(InputType const &a) {
	return a(0,0) * a(1,1) * a(2,2)
		+ a(0,1) * a(1,2) * a(2,0)
		+ a(0,2) * a(1,0) * a(2,1)
		- a(2,0) * a(1,1) * a(0,2)
		- a(2,1) * a(1,2) * a(0,0)
		- a(2,2) * a(1,0) * a(0,1);
}

//I know, it's not a legitimate unit test.
// but maybe some day it will become one ...

void test_v1_tensors() {
	using Real = double;
	using Vector = Tensor::Tensor<Real,Tensor::Upper<3>>;
	
	Vector v = {1,2,3};
	
	TEST_EQ(v.rank, 1);
	TEST_EQ(v.size()(0), 3);
	std::cout << "v^i = " << v << std::endl;

	using OneForm = Tensor::Tensor<Real,Tensor::Lower<3>>;
	OneForm w;
	TEST_EQ(w.rank, 1);
	TEST_EQ(w.size()(0), 3);
	std::cout << "w_i = " << w << std::endl;

	using Metric = Tensor::Tensor<Real,Tensor::Symmetric<Tensor::Lower<3>,Tensor::Lower<3>>>;
	Metric g;
	TEST_EQ(g.rank, 2);
	TEST_EQ(g.size()(0), 3);
	TEST_EQ(g.size()(1), 3);
	for (int i = 0; i < 3; ++i) {
		g(i,i) = 1;
	}
	std::cout << "g_ij = " << g << std::endl;

	using Matrix = Tensor::Tensor<Real,Tensor::Upper<3>,Tensor::Upper<3>>;
	Matrix h;
	TEST_EQ(h.rank, 2);
	TEST_EQ(h.size()(0), 3);
	TEST_EQ(h.size()(1), 3);
	int index = 0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			h(i,j) = ++index;
		}
	}
	std::cout << "h^ij = " << h << std::endl;

	//iterator access
	int j = 0;
	Tensor::Tensor<Real, Tensor::Upper<3>, Tensor::Upper<3>, Tensor::Upper<3>> ta;
	for (auto i = ta.begin(); i != ta.end(); ++i) {
		*i = j++;
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				TEST_EQ(ta(i,j,k), i + 3 * (j + 3 * k));
			}
		}
	}

	//subtensor access not working
	//i'll replace it with write iterators and metaprogram assignments
#if 0
	Tensor::Tensor<Real, Tensor::Upper<3>, Tensor::Upper<3>> tb;
	for (Tensor::Tensor<Real, Tensor::Upper<3>, Tensor::Upper<3>>::iterator i = tb.begin(); i != tb.end(); ++i) *i = 2.;
	std::cout << "tb = " << tb << std::endl;
	ta(0) = tb;
	std::cout << "ta = " << ta << std::endl;
	Tensor::Tensor<Real, Tensor::Upper<3>> tc;
	for (Tensor::Tensor<Real, Tensor::Upper<3>>::iterator i = tc.begin(); i != tc.end(); ++i) *i = 3.;
	std::cout << "tc = " << tc << std::endl;
	//ta(0,0) = tc;		//not working
	ta(0)(0) = tc;
	std::cout << "ta = " << ta << std::endl;
#endif

	//inverse
	Matrix m;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			m(i,j) = i == j ? 1 : 0;
		}
	}

	std::cout << "m: " << m << std::endl;
	std::cout << "determinant: " << determinant33<Matrix>(m) << std::endl;
	

#if 0	//not yet working
	using RiemannTensor = Tensor::Tensor<Real,
		antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>,
		antisymmetric<Tensor::Lower<2>, Tensor::Lower<2>>
	>;
	using RiemannTensorStats = RiemannTensor::TensorStats;
	using RiemannTensorStatsInnerType = RiemannTensorStats::InnerType;
	using RiemannTensorStatsInnerBodyType = RiemannTensorStatsInnerType::BodyType;
	RiemannTensor r;
	std::cout << "size " << r.size() << std::endl;
	std::cout << "rank " << r.rank << std::endl;
	int e = 0;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < i; ++j) {
			for (int k = 0; k < 2; ++k) {
				for (int l = 0; l < k; ++l) {
					((RiemannTensorStatsInnerBodyType&)(r.body(i,j)))(k,l) = ++e;
				}
			}
		}
	}
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				for (int l = 0; l < 2; ++l) {
					std::cout << "r_" << i << j << k << l << " = " << ((RiemannTensorStatsInnerBodyType&)(r.body(i,j)))(k,l) << std::endl;
				}
			}
		}
	}
#endif
}

void test_v1_basics() {
	test_v1_vectors();
	test_v1_tensors();
}
