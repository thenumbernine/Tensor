#include "TensorMath/Tensor.h"
//#include "TensorMath/Inverse.h"
#include "defs.h"
#include <iostream>

template<typename InputType>
typename InputType::Type determinant33(const InputType &a) {
	return a(0,0) * a(1,1) * a(2,2)
		+ a(0,1) * a(1,2) * a(2,0)
		+ a(0,2) * a(1,0) * a(2,1)
		- a(2,0) * a(1,1) * a(0,2)
		- a(2,1) * a(1,2) * a(0,0)
		- a(2,2) * a(1,0) * a(0,1);
}

using namespace std;

//I know, it's not a legitimate unit test.
// but maybe some day it will become one ...

void test_tensors() {
	typedef double Real;
	typedef Tensor<Real,Upper<3>> Vector;
	Vector v;
	TEST_EQ(v.rank, 1);
	TEST_EQ(v.size(), 3);
	std::cout << "v^i = " << v << std::endl;

	typedef Tensor<Real,Lower<3>> OneForm;
	OneForm w;
	TEST_EQ(w.rank, 1);
	TEST_EQ(w.size(), 3);
	std::cout << "w_i = " << w << std::endl;

	typedef Tensor<Real,Symmetric<Lower<3>,Lower<3>>> Metric;
	Metric g;
	TEST_EQ(g.rank, 2);
	TEST_EQ(g.size()(0), 3);
	TEST_EQ(g.size()(1), 3);
	for (int i = 0; i < 3; ++i) {
		g(i,i) = 1;
	}
	std::cout << "g_ij = " << g << std::endl;

	typedef Tensor<Real,Upper<3>,Upper<3>> Matrix;
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
	Tensor<Real, Upper<3>, Upper<3>, Upper<3>> ta;
	for (Tensor<Real, Upper<3>, Upper<3>, Upper<3>>::iterator i = ta.begin(); 
		i != ta.end(); ++i) 
	{
		*i = j++;
	}
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				TEST_EQ(ta(i,j,k), i + 3 * (j + 3 * k));
			}
		}
	}

	//subtensor access not workign
	//i'll replace it with write iterators and metaprogram assignments
	#if 0
	Tensor<Real, Upper<3>, Upper<3>> tb;
	for (Tensor<Real, Upper<3>, Upper<3>>::iterator i = tb.begin(); i != tb.end(); ++i) *i = 2.;
	cout << "tb = " << tb << endl;
	ta(0) = tb;
	cout << "ta = " << ta << endl;
	Tensor<Real, Upper<3>> tc;
	for (Tensor<Real, Upper<3>>::iterator i = tc.begin(); i != tc.end(); ++i) *i = 3.;
	cout << "tc = " << tc << endl;
	//ta(0,0) = tc;		//not working
	ta(0)(0) = tc;
	cout << "ta = " << ta << endl;
	#endif

	//inverse
	Matrix m;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			m(i,j) = i == j ? 1 : 0;
		}
	}

	cout << "m: " << m << endl;
	cout << "determinant: " << determinant33<Matrix>(m) << endl;
	

#if 0	//not yet working
	typedef Tensor<Real, 
		antisymmetric<Lower<2>, Lower<2>>,
		antisymmetric<Lower<2>, Lower<2>>
	> RiemannTensor;
	typedef RiemannTensor::TensorStats RiemannTensorStats;
	typedef RiemannTensorStats::InnerType RiemannTensorStatsInnerType;
	typedef RiemannTensorStatsInnerType::BodyType RiemannTensorStatsInnerBodyType;
	RiemannTensor r;
	cout << "size " << r.size() << endl;
	cout << "rank " << r.rank << endl;
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
					cout << "r_" << i << j << k << l << " = " << ((RiemannTensorStatsInnerBodyType&)(r.body(i,j)))(k,l) << endl;
				}
			}
		}
	}
#endif
}

int main() {
	test_tensors();
}

