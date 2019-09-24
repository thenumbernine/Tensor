#pragma once

/*
inverse function
for any sort of rank-2 Tensor::Tensor object
*/

#include "Tensor/Tensor.h"

namespace Tensor {

template<typename Real>
Real det22(
	Real a00, Real a01, 
	Real a10, Real a11) 
{
	return a00 * a11 - a10 * a01;
}

//I have to specialize determinant by rank
//which means (since rank is an enum rather than a template parameter)
// that I might have to specialize it per-index
// (or make use of static conditions)
template<typename InputType>
struct DeterminantClass;

template<typename Real>
struct DeterminantClass<Tensor<Real, Lower<1>, Lower<1>>> {
	using OutputType = Real;
	using InputType = Tensor<Real, Lower<1>, Lower<1>>;
	OutputType operator()(const InputType &a) const { 
		return a(0,0);
	}
};


template<typename Real>
struct DeterminantClass<Tensor<Real, Symmetric<Lower<1>, Lower<1>>>> {
	using OutputType = Real;
	using InputType = Tensor<Real, Symmetric<Lower<1>, Lower<1>>>;
	OutputType operator()(const InputType &a) const { 
		return a(0,0);
	}
};

template<typename Real>
struct DeterminantClass<Tensor<Real, Lower<2>, Lower<2>>> {
	using OutputType = Real;
	using InputType = Tensor<Real, Lower<2>, Lower<2>>;
	OutputType operator()(const InputType &a) const { 
		return det22(a(0,0), a(0,1), a(1,0), a(1,1));
	}
};

template<typename Real>
struct DeterminantClass<Tensor<Real, Symmetric<Lower<2>, Lower<2>>>> {
	using OutputType = Real;
	using InputType = Tensor<Real, Symmetric<Lower<2>, Lower<2>>>;
	OutputType operator()(const InputType &a) const { 
		return det22(a(0,0), a(0,1), a(1,0), a(1,1));
	}
};

template<typename OutputType, typename InputType>
OutputType determinant33(const InputType &a) {
	return a(0,0) * a(1,1) * a(2,2)
		+ a(0,1) * a(1,2) * a(2,0)
		+ a(0,2) * a(1,0) * a(2,1)
		- a(0,2) * a(1,1) * a(2,0)
		- a(0,1) * a(1,0) * a(2,2)
		- a(0,0) * a(1,2) * a(2,1);
}

//this is where boost::bind would be helpful: for a generic dimension determinant function
template<typename Real>
struct DeterminantClass<Tensor<Real, Lower<3>, Lower<3>>> {
	using OutputType = Real;
	using InputType = Tensor<Real, Lower<3>, Lower<3>>;
	OutputType operator()(const InputType &a) const { 
		return determinant33<OutputType, InputType>(a);
	}
};

template<typename Real>
struct DeterminantClass<Tensor<Real, Symmetric<Lower<3>, Lower<3>>>> {
	using OutputType = Real;
	using InputType = Tensor<Real, Symmetric<Lower<3>, Lower<3>>>;
	OutputType operator()(const InputType &a) const { 
		return determinant33<OutputType, InputType>(a);
	}
};

template<typename InputType>
typename DeterminantClass<InputType>::OutputType
determinant(const InputType &a) {
	return DeterminantClass<InputType>()(a);
}

//currently only used for (gamma^ij) = (gamma_ij)^-1
//so with the still uncertain decision of how to expose / parameterize lowers and uppers of both symmetric and nonsymmetric indexes
// (I was thinking to allow this only for lowers, but of both symmetric and non-symmetric)
//instead I'll just write the specialization.
//another perk of this is that symmetric needs less operations.
// though that could be incorporated into a single function if the tensor iterator returned both the index and the dereference, 
// and we subsequently used the Levi Civita definition in compile-time to calculate the inverse
template<typename InputType>
struct InverseClass;

template<typename Real_>
struct InverseClass<Tensor<Real_, Lower<1>, Lower<1>>> {
	using Real = Real_;
	using InputType = Tensor<Real, Lower<1>, Lower<1>>;
	using OutputType = Tensor<Real, Upper<1>, Upper<1>>;
	OutputType operator()(const InputType &a, const Real &det) const {
		OutputType result;
		result(0,0) = 1. / det;
		return result;
	}
};

template<typename Real_>
struct InverseClass<Tensor<Real_, Symmetric<Lower<1>, Lower<1>>>> {
	using Real = Real_;
	using InputType = Tensor<Real, Symmetric<Lower<1>, Lower<1>>>;
	using OutputType = Tensor<Real, Symmetric<Upper<1>, Upper<1>>>;
	OutputType operator()(const InputType &a, const Real &det) const {
		OutputType result;
		result(0,0) = 1. / det;
		return result;
	}
};

template<typename Real_>
struct InverseClass<Tensor<Real_, Lower<2>, Lower<2>>> {
	using Real = Real_;
	using InputType = Tensor<Real, Lower<2>, Lower<2>>;
	using OutputType = Tensor<Real, Upper<2>, Upper<2>>;
	OutputType operator()(const InputType &a, const Real &det) const {
		OutputType result;
		result(0,0) = a(1,1) / det;
		result(1,0) = -a(1,0) / det;
		result(0,1) = -a(0,1) / det;
		result(1,1) = a(0,0) / det;
		return result;
	}
};

template<typename Real_>
struct InverseClass<Tensor<Real_, Symmetric<Lower<2>, Lower<2>>>> {
	using Real = Real_;
	using InputType = Tensor<Real, Symmetric<Lower<2>, Lower<2>>>;
	using OutputType = Tensor<Real, Symmetric<Upper<2>, Upper<2>>>;
	OutputType operator()(const InputType &a, const Real &det) const {
		OutputType result;
		result(0,0) = a(1,1) / det;
		result(1,0) = -a(1,0) / det;
		//result(0,1) = -a(0,1) / det;	//<- symmetric only needs one
		result(1,1) = a(0,0) / det;
		return result;
	}
};

template<typename Real_>
struct InverseClass<Tensor<Real_, Lower<3>, Lower<3>>> {
	using Real = Real_;
	using InputType = Tensor<Real, Lower<3>, Lower<3>>;
	using OutputType = Tensor<Real, Upper<3>, Upper<3>>;
	OutputType operator()(const InputType &a, const Real &det) const {
		OutputType result;
		Real invDet = Real(1) / det;
		for (int j = 0; j < 3; ++j) {
			int j1 = (j + 1) % 3;
			int j2 = (j + 2) % 3;
			for (int i = 0; i < 3; ++i) {
				int i1 = (i + 1) % 3;
				int i2 = (i + 2) % 3;
				result(j,i) = invDet * (a(i1,j1) * a(i2,j2) - a(i2,j1) * a(i1,j2));
			}
		}
		return result;
	}
};

template<typename Real_>
struct InverseClass<Tensor<Real_, Symmetric<Lower<3>, Lower<3>>>> {
	using Real = Real_;
	using InputType = Tensor<Real, Symmetric<Lower<3>, Lower<3>>>;
	using OutputType = Tensor<Real, Symmetric<Upper<3>, Upper<3>>>;
	OutputType operator()(const InputType &a, const Real &det) const {
		OutputType result;
		//symmetric, so only do Lower triangular
		result(0,0) = det22(a(1,1), a(1,2), a(2,1), a(2,2)) / det;
		result(1,0) = det22(a(1,2), a(1,0), a(2,2), a(2,0)) / det;
		result(1,1) = det22(a(0,0), a(0,2), a(2,0), a(2,2)) / det;
		result(2,0) = det22(a(1,0), a(1,1), a(2,0), a(2,1)) / det;
		result(2,1) = det22(a(0,1), a(0,0), a(2,1), a(2,0)) / det;
		result(2,2) = det22(a(0,0), a(0,1), a(1,0), a(1,1)) / det;
		return result;
	}
};

template<typename InputType>
typename InverseClass<InputType>::OutputType inverse(const InputType &a, const typename InverseClass<InputType>::Real &det) {
	return InverseClass<InputType>()(a, det);
}

template<typename InputType>
typename InverseClass<InputType>::OutputType inverse(const InputType &a) {
	return InverseClass<InputType>()(a, determinant(a));
}

}
