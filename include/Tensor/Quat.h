#pragma once

#include "Tensor/Vector.h"
#include <cmath>

// TODO any vec& member return types will have to be overloaded

namespace Tensor {

template<typename T>
struct _quat : public _vec4<T> {
	using This = _quat;
	using Super = _vec4<T>;
	using vec3 = _vec3<T>;

	_quat() : Super(0,0,0,1) {}
	_quat(T const & x, T const & y, T const & z, T const & w) 
	: Super(x,y,z,w) {}

	//conjugate assuming the quat is unit length
	_quat unitConj() const {
		return _quat(-(*this)(0), -(*this)(1), -(*this)(2), (*this)(3));
	}

	//conjugate
	_quat conj() const {
		return unitConj() / Super::lenSq();
	}

	//angle-axis, where angle is in radians
	_quat fromAngleAxis() const {
		T const c = cos((*this)(3) / 2);
		T const n = sqrt((*this)(0) * (*this)(0) + (*this)(1) * (*this)(1) + (*this)(2) * (*this)(2));
		T const sn = sin((*this)(3) / 2) / n;
		return {sn * (*this)(0), sn * (*this)(1), sn * (*this)(2), c};
	}

	static float angleAxisEpsilon;

	_quat toAngleAxis() const {
		T const cosHalfAngle = (*this)(3);
		if ((*this)(3) < -1.) (*this)(3) = -1.;
		if ((*this)(3) > 1.) (*this)(3) = 1.;
		T const halfAngle = acos(cosHalfAngle);
		T const scale = sin(halfAngle);

		if (scale >= -angleAxisEpsilon && scale <= angleAxisEpsilon) {
			return _quat(0,0,1,0);
		}

		T const invscale = 1. / scale;
		return {
			(*this)(0) * invscale,
			(*this)(1) * invscale,
			(*this)(2) * invscale,
			2. * halfAngle,
		};
	}

	static _quat mul(_quat const &q, _quat const &r) {
		T const a = (q(3) + q(0)) * (r(3) + r(0));
		T const b = (q(2) - q(1)) * (r(1) - r(2));
		T const c = (q(0) - q(3)) * (r(1) + r(2));
		T const d = (q(1) + q(2)) * (r(0) - r(3));
		T const e = (q(0) + q(2)) * (r(0) + r(1));
		T const f = (q(0) - q(2)) * (r(0) - r(1));
		T const g = (q(3) + q(1)) * (r(3) - r(2));
		T const h = (q(3) - q(1)) * (r(3) + r(2));

		return { 
			 a - ( e + f + g + h) / 2,
			-c + ( e - f + g - h) / 2,
			-d + ( e - f - g + h) / 2,
			 b + (-e - f + g + h) / 2,
		};
	}

	vec3 rotate(vec3 const & v) const {
		_quat vq = {v(0), v(1), v(2)};
		vq = *this * vq * unitConj();
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

template<typename T>
float _quat<T>::angleAxisEpsilon = 1e-4;

template<typename T>
_quat<T> operator*(_quat<T> a, _quat<T> b) {
	return _quat<T>::mul(a,b);
}

using quati = _quat<int>;	// I don't judge
using quatf = _quat<float>;
using quatd = _quat<double>;

}
