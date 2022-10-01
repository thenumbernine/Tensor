#pragma once

#include "Tensor/Vector.h"
#include "Tensor/clamp.h"
#include <cmath>

// TODO any vec& member return types will have to be overloaded

namespace Tensor {

template<typename T>
struct _quat : public _vec4<T> {
	using Super = _vec4<T>;
	using This = _quat;
	TENSOR_VECTOR_HEADER(4)
	TENSOR_HEADER()
	using vec3 = _vec3<T>;

	constexpr _quat() : Super(0,0,0,1) {}
	constexpr _quat(T const & w) : Super(0,0,0,w) {}
	constexpr _quat(T const & x, T const & y, T const & z, T const & w) : Super(x,y,z,w) {}
	TENSOR_ADD_DIMS()	// needed by TENSOR_ADD_CTOR_FOR_GENERIC_TENSORS
	TENSOR_ADD_CTOR_FOR_GENERIC_TENSORS(_quat, _vec)
	TENSOR_ADD_LAMBDA_CTOR(_quat)
	TENSOR_ADD_ITERATOR()

	//conjugate 
	// same as inverse if the quat is unit length
	_quat conjugate() const {
		return _quat(-this->x, -this->y, -this->z, this->w);
	}

	//inverse
	_quat inverse() const {
		return conjugate() / this->lenSq();
	}

	//angle-axis, where angle is in radians
	_quat fromAngleAxis() const {
		T const c = cos(this->w / 2);
		T const n = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
		T const sn = sin(this->w / 2) / n;
		return {sn * this->x, sn * this->y, sn * this->z, c};
	}

	static T angleAxisEpsilon;

	_quat toAngleAxis() const {
		T const cosHalfAngle = clamp(this->w, (T)-1, (T)1);
		T const halfAngle = acos(cosHalfAngle);
		T const scale = sin(halfAngle);
		if (std::abs(scale) <= angleAxisEpsilon) return _quat(0,0,1,0);
		return {this->x / scale, this->y / scale, this->z / scale, 2 * halfAngle};
	}

	static _quat mul(_quat const &q, _quat const &r) {
		T const a = (q.w + q.x) * (r.w + r.x);
		T const b = (q.z - q.y) * (r.y - r.z);
		T const c = (q.x - q.w) * (r.y + r.z);
		T const d = (q.y + q.z) * (r.x - r.w);
		T const e = (q.x + q.z) * (r.x + r.y);
		T const f = (q.x - q.z) * (r.x - r.y);
		T const g = (q.w + q.y) * (r.w - r.z);
		T const h = (q.w - q.y) * (r.w + r.z);

		return {
			 a - ( e + f + g + h) / 2,
			-c + ( e - f + g - h) / 2,
			-d + ( e - f - g + h) / 2,
			 b + (-e - f + g + h) / 2,
		};
	}

	vec3 rotate(vec3 const & v) const {
		return *this * _quat(v) * conjugate();
	}

	vec3 xAxis() const {
		return {
			1 - 2 * (this->y * this->y + this->z * this->z),
			2 * (this->x * this->y + this->z * this->w),
			2 * (this->x * this->z - this->w * this->y)
		};
	}

	vec3 yAxis() const {
		return {
			2 * (this->x * this->y - this->w * this->z),
			1 - 2 * (this->x * this->x + this->z * this->z),
			2 * (this->y * this->z + this->w * this->x)
		};
	}

	vec3 zAxis() const {
		return {
			2 * (this->x * this->z + this->w * this->y),
			2 * (this->y * this->z - this->w * this->x),
			1 - 2 * (this->x * this->x + this->y * this->y)
		};
	}
};

template<typename T>
T _quat<T>::angleAxisEpsilon = (T)1e-4;

// TODO more math operators, correctly implementing quaternion math (ex: scalar mul, quat inv)
template<typename T>
_quat<T> operator*(_quat<T> a, _quat<T> b) {
	return _quat<T>::mul(a,b);
}

template<typename T>
_quat<T> normalize(_quat<T> const & q) {
	T len = q.length();
	if (len <= _quat<T>::angleAxisEpsilon) return _quat<T>();
	return (_quat<T>)(q / len);
}

using quati = _quat<int>;	// I don't judge
using quatf = _quat<float>;
using quatd = _quat<double>;

}
