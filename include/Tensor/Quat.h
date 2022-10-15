#pragma once

#include "Tensor/Vector.h"
#include "Tensor/clamp.h"
#include <cmath>

// TODO any vec& member return types, like operator+=, will have to be overloaded
//  and it looks like I can't crtp the operator+= members due to _vec having unions?

namespace Tensor {

#define TENSOR_HEADER_QUAT(classname, Inner_)\
	TENSOR_THIS(classname)\
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, 4, 1)\
	TENSOR_TEMPLATE_T(classname)\
	TENSOR_HEADER_VECTOR_SPECIFIC() /* defines localCount=localDim, matching for _vec and _quat */\
	TENSOR_HEADER()

template<typename Inner_>
struct _quat;

template<typename T> struct is_quat : public std::false_type {};
template<typename T> struct is_quat<_quat<T>> : public std::true_type {};
template<typename T> constexpr bool is_quat_v = is_quat<T>::value;

template<typename T>
_quat<T> operator*(_quat<T> a, _quat<T> b);

template<typename Inner_>
struct _quat : public _vec4<Inner_> {
	using Super = _vec4<Inner_>;
	TENSOR_HEADER_QUAT(_quat, Inner_)
	
	using vec3 = _vec3<Inner>;

	constexpr _quat() : Super(0,0,0,1) {}
	
	// mathematically, a real is equivalent to a quaternion with only the real component defined ...
	constexpr _quat(Inner const & w) : Super(0,0,0,w) {}
	
	constexpr _quat(Inner const & x, Inner const & y, Inner const & z, Inner const & w) : Super(x,y,z,w) {}

	//TENSOR_ADD_OPS parts:
	TENSOR_ADD_CTOR_FOR_GENERIC_TENSORS(_quat)
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
		Inner const c = cos(this->w / 2);
		Inner const n = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
		Inner const sn = sin(this->w / 2) / n;
		return {sn * this->x, sn * this->y, sn * this->z, c};
	}

	static Inner angleAxisEpsilon;

	_quat toAngleAxis() const {
		Inner const cosHalfAngle = clamp(this->w, (Inner)-1, (Inner)1);
		Inner const halfAngle = acos(cosHalfAngle);
		Inner const scale = sin(halfAngle);
		if (std::abs(scale) <= angleAxisEpsilon) return _quat(0,0,1,0);
		return {this->x / scale, this->y / scale, this->z / scale, 2 * halfAngle};
	}

	static _quat mul(_quat const &q, _quat const &r) {
		Inner const a = (q.w + q.x) * (r.w + r.x);
		Inner const b = (q.z - q.y) * (r.y - r.z);
		Inner const c = (q.x - q.w) * (r.y + r.z);
		Inner const d = (q.y + q.z) * (r.x - r.w);
		Inner const e = (q.x + q.z) * (r.x + r.y);
		Inner const f = (q.x - q.z) * (r.x - r.y);
		Inner const g = (q.w + q.y) * (r.w - r.z);
		Inner const h = (q.w - q.y) * (r.w + r.z);

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

	_quat & operator*=(_quat const & o) {
		*this = *this * o;
		return *this;
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
