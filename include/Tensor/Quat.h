#pragma once

#include "Tensor/Vector.h"
#include <cmath>

namespace Tensor {

template<typename Type>
struct Quat : public GenericVector<Type, 4, Type, Quat<Type>> {
	using Super = GenericVector<Type, 4, Type, Quat<Type>>;
	using vec3 = Vector<Type, 3>;

	Quat() : Super(0,0,0,1) {}
	Quat(Type x, Type y, Type z, Type w) : Super(x,y,z,w) {}

	//conjugate assuming the quat is unit length
	Quat unitConj() const {
		return Quat(-(*this)(0), -(*this)(1), -(*this)(2), (*this)(3));
	}

	Quat fromAngleAxis() {
		Type c = cos((*this)(3) * .5);
		Type n = sqrt((*this)(0) * (*this)(0) + (*this)(1) * (*this)(1) + (*this)(2) * (*this)(2));
		Type sn = sin((*this)(3) * .5) / n;
		return Quat(sn * (*this)(0), sn * (*this)(1), sn * (*this)(2), c);
	}
	
	Quat toAngleAxis() {
		Type cosHalfAngle = (*this)(3);
		if ((*this)(3) < -1.) (*this)(3) = -1.;
		if ((*this)(3) > 1.) (*this)(3) = 1.;
		Type halfAngle = acos(cosHalfAngle);
		Type scale = sin(halfAngle);

		if (scale >= -1e-4 && scale <= 1e-4) {
			return Quat(0,0,1,0);
		}

		scale = 1. / scale;
		return Quat(
			(*this)(0) * scale,
			(*this)(1) * scale,
			(*this)(2) * scale,
			2. * halfAngle);
	}

	static Quat mul(Quat const &q, Quat const &r) {
		Type a = (q(3) + q(0)) * (r(3) + r(0));
		Type b = (q(2) - q(1)) * (r(1) - r(2));
		Type c = (q(0) - q(3)) * (r(1) + r(2));
		Type d = (q(1) + q(2)) * (r(0) - r(3));
		Type e = (q(0) + q(2)) * (r(0) + r(1));
		Type f = (q(0) - q(2)) * (r(0) - r(1));
		Type g = (q(3) + q(1)) * (r(3) - r(2));
		Type h = (q(3) - q(1)) * (r(3) + r(2));

		return Quat(
			a - .5 * ( e + f + g + h),
			-c + .5 * ( e - f + g - h),
			-d + .5 * ( e - f - g + h),
			b + .5 * (-e - f + g + h));
	}

	vec3 rotate(vec3 const & v) const {
		Quat vq = {v(0), v(1), v(2), 0};
		vq = mul(mul(*this, vq), unitConj());
		return {vq(0), vq(1), vq(2)};
	}

	vec3 xAxis() const {
		return {
			1 - 2 * ((*this)(1) * (*this)(1) + (*this)(2) * (*this)(2)),
			2 * ((*this)(0) * (*this)(1) + (*this)(2) * (*this)(3)),
			2 * ((*this)(0) * (*this)(2) - (*this)(3) * (*this)(1))
		};
	}

	vec3 yAxis() const {
		return {
			2 * ((*this)(0) * (*this)(1) - (*this)(3) * (*this)(2)),
			1 - 2 * ((*this)(0) * (*this)(0) + (*this)(2) * (*this)(2)),
			2 * ((*this)(1) * (*this)(2) + (*this)(3) * (*this)(0))
		};
	}

	vec3 zAxis() const {
		return {
			2 * ((*this)(0) * (*this)(2) + (*this)(3) * (*this)(1)),
			2 * ((*this)(1) * (*this)(2) - (*this)(3) * (*this)(0)),
			1 - 2 * ((*this)(0) * (*this)(0) + (*this)(1) * (*this)(1))
		};
	}
};

template<typename Type>
Quat<Type> operator*(Quat<Type> a, Quat<Type> b) {
	return Quat<Type>::mul(a,b);
}

}
