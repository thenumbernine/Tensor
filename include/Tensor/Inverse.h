#pragma once

//atm Vector.h includes Inverse.h so this is moot:
#include "Tensor/Vector.h.h"
#include "Tensor/Inverse.h.h"

namespace Tensor {

// determinant for matrix
// TODO generalize or at least expose Levi-Civita tensor as constexpr 
// for implementing cross in higher dimensions, determinant, and whatever else.

template<typename T>
T det22elem(T const & a00, T const & a01, T const & a10, T const & a11) {
	return a00 * a11 - a01 * a10;
}

template<typename T>
typename T::Scalar determinant22(T const & a) {
	return det22elem(a(0,0), a(0,1), a(1,0), a(1,1));
}

template<typename T>
typename T::Scalar determinant33(T const & a) {
	return a(0,0) * a(1,1) * a(2,2)
		+ a(0,1) * a(1,2) * a(2,0)
		+ a(0,2) * a(1,0) * a(2,1)
		- a(0,2) * a(1,1) * a(2,0)
		- a(0,1) * a(1,0) * a(2,2)
		- a(0,0) * a(1,2) * a(2,1);
}

template<typename M>
typename M::Scalar determinant44(M const & a) {
	using T = typename M::Scalar;
	//autogen'd with symmath
	T const tmp1 = a(2,2) * a(3,3);
	T const tmp2 = a(2,3) * a(3,2);
	T const tmp3 = a(2,1) * a(3,3);
	T const tmp4 = a(2,3) * a(3,1);
	T const tmp5 = a(2,1) * a(3,2);
	T const tmp6 = a(2,2) * a(3,1);
	T const tmp7 = a(2,0) * a(3,3);
	T const tmp8 = a(2,3) * a(3,0);
	T const tmp9 = a(2,0) * a(3,2);
	T const tmp10 = a(2,2) * a(3,0);
	T const tmp11 = a(2,0) * a(3,1);
	T const tmp12 = a(2,1) * a(3,0);
	return a(0,0) * a(1,1) * tmp1 
		- a(0,0) * a(1,1) * tmp2 
		- a(0,0) * a(1,2) * tmp3
		+ a(0,0) * a(1,2) * tmp4
		+ a(0,0) * a(1,3) * tmp5 
		- a(0,0) * a(1,3) * tmp6 
		- a(0,1) * a(1,0) * tmp1
		+ a(0,1) * a(1,0) * tmp2
		+ a(0,1) * a(1,2) * tmp7 
		- a(0,1) * a(1,2) * tmp8 
		- a(0,1) * a(1,3) * tmp9
		+ a(0,1) * a(1,3) * tmp10
		+ a(0,2) * a(1,0) * tmp3 
		- a(0,2) * a(1,0) * tmp4 
		- a(0,2) * a(1,1) * tmp7
		+ a(0,2) * a(1,1) * tmp8
		+ a(0,2) * a(1,3) * tmp11 
		- a(0,2) * a(1,3) * tmp12 
		- a(0,3) * a(1,0) * tmp5
		+ a(0,3) * a(1,0) * tmp6
		+ a(0,3) * a(1,1) * tmp9 
		- a(0,3) * a(1,1) * tmp10
		+ a(0,3) * a(1,2) * tmp12 
		- a(0,3) * a(1,2) * tmp11;
}

//matrix specializations

template<typename T>
T determinant(_mat<T,1,1> const & a) {
	return a.x.x;
}

template<typename T>
T determinant(_mat2x2<T> const & a) {
	return determinant22(a);
}

template<typename T>
T determinant(_mat3x3<T> const & a) {
	return determinant33(a);
}

template<typename T>
T determinant(_mat4x4<T> const & a) {
	return determinant44(a);
}

// determinant for symmetric

template<typename T>
T determinant(_sym<T,1> const & a) {
	return a(0,0);
}

template<typename T>
T determinant(_sym2<T> const & a) {
	return determinant22(a);
}

template<typename T>
T determinant(_sym3<T> const & a) {
	return determinant33(a);
}

template<typename T>
T determinant(_sym4<T> const & a) {
	return determinant44(a);
}

template<typename M>
typename M::Scalar determinantNN(M const & a) {
	using T = typename M::Scalar;
	static_assert(M::rank == 2);
	static_assert(M::template dim<0> == M::template dim<1>);
	constexpr int dim = M::template dim<0>;
	T sign = 1;
	T sum = {};
	for (int k = 0; k < dim; ++k) {
		// TODO ReplaceLocalDim / ReplaceDim to preserve symmetry?
		using subM = _mat<T,dim-1,dim-1>;
		subM sub;
		for (int i = 0; i < dim-1; ++i) {
			for (int j = 0; j < dim-1; ++j) {
				sub(i,j) = a(i+1, j + (j>=k));
			}
		}
		sum += sign * a(0,k) * Tensor::determinant(sub);
		sign = -sign;
	}
	return sum;
}

//general case

template<typename T, int dim>
requires(dim>4)
T determinant(_mat<T,dim,dim> const & a) {
	return determinantNN(a);
}

template<typename T, int dim>
requires (dim > 4)
T determinant(_sym<T,dim> const & a) {
	return determinantNN(a);
}

template<typename T>
requires is_tensor_v<T>
typename T::Scalar determinant(T const & a) {
	return determinantNN(a);
}



// inverse for matrix
//  I could write out all specialized functions
//  but then i'd have to fwd-declare them also ...
// So I will just write out the specializations as a 2nd function


template<typename T>
_mat<T,1,1> inverseImpl(_mat<T,1,1> const & a, T const & det) {
	return (T)1 / det;	// == 1 / a.x.x;
}

template<typename T>
_mat2x2<T> inverseImpl(_mat2x2<T> const & a, T const & det) {
	return {
		{
			 a.s1.s1 / det,
			-a.s0.s1 / det,
		},
		{
			-a.s1.s0 / det,
			 a.s0.s0 / det,
		}
	};
}

template<typename T>
_mat3x3<T> inverseImpl(_mat3x3<T> const & a, T const & det) {
	return {
		{
			(-a.s1.s2 * a.s2.s1 +  a.s1.s1 * a.s2.s2) / det,
			( a.s0.s2 * a.s2.s1 + -a.s0.s1 * a.s2.s2) / det,
			(-a.s0.s2 * a.s1.s1 +  a.s0.s1 * a.s1.s2) / det,
		}, {
			( a.s1.s2 * a.s2.s0 + -a.s1.s0 * a.s2.s2) / det,
			(-a.s0.s2 * a.s2.s0 +  a.s0.s0 * a.s2.s2) / det,
			( a.s0.s2 * a.s1.s0 + -a.s0.s0 * a.s1.s2) / det,
		}, {
			(-a.s1.s1 * a.s2.s0 +  a.s1.s0 * a.s2.s1) / det,
			( a.s0.s1 * a.s2.s0 + -a.s0.s0 * a.s2.s1) / det,
			(-a.s0.s1 * a.s1.s0 +  a.s0.s0 * a.s1.s1) / det,
		}
	};
}

//from : https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
template<typename T>
_mat4x4<T> inverseImpl(_mat4x4<T> const & a, T const & det) {
	T const a2323 = a.s2.s2 * a.s3.s3 - a.s2.s3 * a.s3.s2;
	T const a1323 = a.s2.s1 * a.s3.s3 - a.s2.s3 * a.s3.s1;
	T const a1223 = a.s2.s1 * a.s3.s2 - a.s2.s2 * a.s3.s1;
	T const a0323 = a.s2.s0 * a.s3.s3 - a.s2.s3 * a.s3.s0;
	T const a0223 = a.s2.s0 * a.s3.s2 - a.s2.s2 * a.s3.s0;
	T const a0123 = a.s2.s0 * a.s3.s1 - a.s2.s1 * a.s3.s0;
	T const a2313 = a.s1.s2 * a.s3.s3 - a.s1.s3 * a.s3.s2;
	T const a1313 = a.s1.s1 * a.s3.s3 - a.s1.s3 * a.s3.s1;
	T const a1213 = a.s1.s1 * a.s3.s2 - a.s1.s2 * a.s3.s1;
	T const a2312 = a.s1.s2 * a.s2.s3 - a.s1.s3 * a.s2.s2;
	T const a1312 = a.s1.s1 * a.s2.s3 - a.s1.s3 * a.s2.s1;
	T const a1212 = a.s1.s1 * a.s2.s2 - a.s1.s2 * a.s2.s1;
	T const a0313 = a.s1.s0 * a.s3.s3 - a.s1.s3 * a.s3.s0;
	T const a0213 = a.s1.s0 * a.s3.s2 - a.s1.s2 * a.s3.s0;
	T const a0312 = a.s1.s0 * a.s2.s3 - a.s1.s3 * a.s2.s0;
	T const a0212 = a.s1.s0 * a.s2.s2 - a.s1.s2 * a.s2.s0;
	T const a0113 = a.s1.s0 * a.s3.s1 - a.s1.s1 * a.s3.s0;
	T const a0112 = a.s1.s0 * a.s2.s1 - a.s1.s1 * a.s2.s0;
	return { 
		{
		    (a.s1.s1 * a2323 - a.s1.s2 * a1323 + a.s1.s3 * a1223) / det,
		   -(a.s0.s1 * a2323 - a.s0.s2 * a1323 + a.s0.s3 * a1223) / det,
		    (a.s0.s1 * a2313 - a.s0.s2 * a1313 + a.s0.s3 * a1213) / det,
		   -(a.s0.s1 * a2312 - a.s0.s2 * a1312 + a.s0.s3 * a1212) / det,
		}, {
		   -(a.s1.s0 * a2323 - a.s1.s2 * a0323 + a.s1.s3 * a0223) / det,
		    (a.s0.s0 * a2323 - a.s0.s2 * a0323 + a.s0.s3 * a0223) / det,
		   -(a.s0.s0 * a2313 - a.s0.s2 * a0313 + a.s0.s3 * a0213) / det,
		    (a.s0.s0 * a2312 - a.s0.s2 * a0312 + a.s0.s3 * a0212) / det,
		}, {
		    (a.s1.s0 * a1323 - a.s1.s1 * a0323 + a.s1.s3 * a0123) / det,
		   -(a.s0.s0 * a1323 - a.s0.s1 * a0323 + a.s0.s3 * a0123) / det,
		    (a.s0.s0 * a1313 - a.s0.s1 * a0313 + a.s0.s3 * a0113) / det,
		   -(a.s0.s0 * a1312 - a.s0.s1 * a0312 + a.s0.s3 * a0112) / det,
		}, {
		   -(a.s1.s0 * a1223 - a.s1.s1 * a0223 + a.s1.s2 * a0123) / det,
		    (a.s0.s0 * a1223 - a.s0.s1 * a0223 + a.s0.s2 * a0123) / det,
		   -(a.s0.s0 * a1213 - a.s0.s1 * a0213 + a.s0.s2 * a0113) / det,
		    (a.s0.s0 * a1212 - a.s0.s1 * a0212 + a.s0.s2 * a0112) / det,
		}
	};
}

// inverseImpl for symmetric
// has a different # of written fields so might as well optimize for it

template<typename T>
_sym<T,1> inverseImpl(_sym<T,1> const & a, T const & det) {
	return (T)1 / det;	// == 1 / a.xx;	 ... which to use?
}

// TODO sym2 from mat2 
template<typename T>
_sym2<T> inverseImpl(_sym2<T> const & a, T const & det) {
	return {
		 a(1,1) / det, // AInv(0,0)
		-a(0,1) / det, // AInv(1,0)
		 a(0,0) / det, // AInv(1,1)
	};
}

template<typename T>
_sym3<T> inverseImpl(_sym3<T> const & a, T const & det) {
	return {
		det22elem(a(1,1), a(1,2), a(2,1), a(2,2)) / det,	// AInv(0,0)
		det22elem(a(1,2), a(1,0), a(2,2), a(2,0)) / det,	// AInv(1,0)
		det22elem(a(0,0), a(0,2), a(2,0), a(2,2)) / det,	// AInv(1,1)
		det22elem(a(1,0), a(1,1), a(2,0), a(2,1)) / det,	// AInv(2,0)
		det22elem(a(0,1), a(0,0), a(2,1), a(2,0)) / det,	// AInv(2,1)
		det22elem(a(0,0), a(0,1), a(1,0), a(1,1)) / det,	// AInv(2,2)
	};
}

//from the 4x4 case above, which is from: https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
// hmm after comparing symmetric outputs i'm not seeing the results match, makes me suspicious of the above implemenation ...
template<typename T>
_sym4<T> inverseImpl(_sym4<T> const & a, T const & det) {
	T const a2323 = a.s22 * a.s33 - a.s23 * a.s23;
	T const a1323 = a.s12 * a.s33 - a.s23 * a.s13;
	T const a1223 = a.s12 * a.s23 - a.s22 * a.s13;
	T const a0323 = a.s02 * a.s33 - a.s23 * a.s03;
	T const a0223 = a.s02 * a.s23 - a.s22 * a.s03;
	//T const a0123 = a.s02 * a.s13 - a.s12 * a.s03;
	T const a2313 = a.s12 * a.s33 - a.s13 * a.s23;
	T const a1313 = a.s11 * a.s33 - a.s13 * a.s13;
	T const a1213 = a.s11 * a.s23 - a.s12 * a.s13;
	T const a2312 = a.s12 * a.s23 - a.s13 * a.s22;
	T const a1312 = a.s11 * a.s23 - a.s13 * a.s12;
	T const a1212 = a.s11 * a.s22 - a.s12 * a.s12;
	T const a0313 = a.s01 * a.s33 - a.s13 * a.s03;
	T const a0213 = a.s01 * a.s23 - a.s12 * a.s03;
	T const a0312 = a.s01 * a.s23 - a.s13 * a.s02;
	T const a0212 = a.s01 * a.s22 - a.s12 * a.s02;
	T const a0113 = a.s01 * a.s13 - a.s11 * a.s03;
	T const a0112 = a.s01 * a.s12 - a.s11 * a.s02;
	_sym4<T> result; 
	result(0,0) = (a.s11 * a2323 - a.s12 * a1323 + a.s13 * a1223) / det;
	result(0,1) = -(a.s01 * a2323 - a.s02 * a1323 + a.s03 * a1223) / det;
	//result(1,0) = -(a.s01 * a2323 - a.s12 * a0323 + a.s13 * a0223) / det;
	result(0,2) = (a.s01 * a2313 - a.s02 * a1313 + a.s03 * a1213) / det;
	//result(2,0) = (a.s01 * a1323 - a.s11 * a0323 + a.s13 * a0123) / det;
	result(0,3) = -(a.s01 * a2312 - a.s02 * a1312 + a.s03 * a1212) / det;
	//result(3,0) = -(a.s01 * a1223 - a.s11 * a0223 + a.s12 * a0123) / det;
	result(1,1) = (a.s00 * a2323 - a.s02 * a0323 + a.s03 * a0223) / det;
	result(1,2) = -(a.s00 * a2313 - a.s02 * a0313 + a.s03 * a0213) / det;
	//result(2,1) = -(a.s00 * a1323 - a.s01 * a0323 + a.s03 * a0123) / det;
	result(1,3) = (a.s00 * a2312 - a.s02 * a0312 + a.s03 * a0212) / det;
	//result(3,1) = (a.s00 * a1223 - a.s01 * a0223 + a.s02 * a0123) / det;
	result(2,2) = (a.s00 * a1313 - a.s01 * a0313 + a.s03 * a0113) / det;
	result(2,3) = -(a.s00 * a1312 - a.s01 * a0312 + a.s03 * a0112) / det;
	//result(3,2) = -(a.s00 * a1213 - a.s01 * a0213 + a.s02 * a0113) / det;
	result(3,3) = (a.s00 * a1212 - a.s01 * a0212 + a.s02 * a0112) / det;
	return result;
}

template<typename T>
requires is_tensor_v<T>
T inverse(T const & a, typename T::Scalar const & det) {
	return inverseImpl(a, det);
}

// inverse without determinant

template<typename T>
requires is_tensor_v<T>
T inverse(T const & a) {
	return inverse(a, determinant(a));
}

}
