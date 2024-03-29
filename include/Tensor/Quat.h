#pragma once

#include "Tensor/Vector.h"
#include "Tensor/clamp.h"
#include <cmath>

// TODO any vec& member return types, like operator+=, will have to be overloaded
//  and it looks like I can't crtp the operator+= members due to vec having unions?

namespace Tensor {

template<typename Inner_>
struct quat;

template<typename T> struct is_quat : public std::false_type {};
template<typename T> struct is_quat<quat<T>> : public std::true_type {};
template<typename T> constexpr bool is_quat_v = is_quat<T>::value;

template<typename T>
quat<T> operator*(quat<T> a, quat<T> b);

template<typename Inner_>
struct quat : public vec4<Inner_> {
	using Super = vec4<Inner_>;
	// disable the is_tensor_v flag for quaternion so tensor-mul doesn't try indexing into it, so that a matrix-of-quats times a matrix-of-quats produces a matrix-of-quats (and not a rank-5 object)
	//TENSOR_THIS(quat)
	using This = quat;
	//static constexpr bool isTensorFlag = true;
	static constexpr bool isTensorFlag = false;
	static constexpr bool dontUseFieldsOStream = true;
	
	TENSOR_SET_INNER_LOCALDIM_LOCALRANK(Inner_, 4, 1)
	TENSOR_TEMPLATE_T(quat)
	TENSOR_HEADER_VECTOR_SPECIFIC() // defines localCount=localDim, matching for vec and quat
	TENSOR_HEADER()
	
	using vec3 = ::Tensor::vec3<Inner>;
	using mat3x3 = ::Tensor::mat3x3<Inner>;

// ok here's a dilemma ... 
// default ident quat would be useful for rotations
// but for sums-of-quats, and consistency of equating quats with reals, it is useful to default this to zero
//	constexpr quat() : Super(0,0,0,1) {}
	constexpr quat() : Super(0,0,0,0) {}
	
	// mathematically, a real is equivalent to a quaternion with only the real component defined ...
	constexpr quat(Inner const & w) : Super(0,0,0,w) {}
	
	constexpr quat(Inner const & x, Inner const & y, Inner const & z, Inner const & w) : Super(x,y,z,w) {}

	//TENSOR_ADD_OPS parts:
	// ok I don't just want any tensor constructor ...
	//TENSOR_ADD_CTOR_FOR_GENERIC_TENSORS(quat)
	// for 3-component vectors I want to initialize s[0] to 0 and fill the rest
	// for 4-component vectors I want to copy all across
	template<typename T>
	requires (is_tensor_v<T> && T::rank == 1 && T::template dim<0> == 3)
	quat(T const & t) : Super(t(0), t(1), t(2), 0) {}
	template<typename T>
	requires ((is_tensor_v<T> && T::rank == 1 && T::template dim<0> == 4) || (is_quat_v<T>))
	quat(T const & t) : Super(t(0), t(1), t(2), t(3)) {}

	TENSOR_ADD_LAMBDA_CTOR(quat)
	TENSOR_ADD_ITERATOR()
	
	//conjugate 
	// same as inverse if the quat is unit length
	quat conjugate() const {
		return quat(-this->x, -this->y, -this->z, this->w);
	}

	//inverse
	quat inverse() const {
		return conjugate() / this->lenSq();
	}

	//angle-axis, where angle is in radians
	quat fromAngleAxis() const {
		Inner const c = cos(this->w / 2);
		Inner const n = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
		Inner const sn = sin(this->w / 2) / n;
		return {sn * this->x, sn * this->y, sn * this->z, c};
	}

	static Inner angleAxisEpsilon;

	quat toAngleAxis() const {
		Inner const cosHalfAngle = clamp(this->w, (Inner)-1, (Inner)1);
		Inner const halfAngle = acos(cosHalfAngle);
		Inner const scale = sin(halfAngle);
		if (std::abs(scale) <= angleAxisEpsilon) return quat(0,0,1,0);
		return {this->x / scale, this->y / scale, this->z / scale, 2 * halfAngle};
	}

	static quat mul(quat const &q, quat const &r) {
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

	// would like a shorthand name for this instead of 'template subset<3>' ...
	// .vec and .vec3 are taken
	vec3 & axis() { return Super::template subset<3>(); }
	vec3 const & axis() const { return Super::template subset<3>(); }

	vec3 rotate(vec3 const & v) const {
		return (*this * quat(v) * conjugate()).axis();
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

	mat3x3 toMatrix() const {
		return mat3x3(xAxis(), yAxis(), zAxis()).transpose();
	}

	quat & operator*=(quat const & o) {
		*this = *this * o;
		return *this;
	}
	
	quat operator-() const {
		return quat(Super::operator-());
	}
};

template<typename T>
T quat<T>::angleAxisEpsilon = (T)1e-4;

// TODO more math operators, correctly implementing quaternion math (ex: scalar mul, quat inv)
template<typename T>
quat<T> operator*(quat<T> a, quat<T> b) {
	return quat<T>::mul(a,b);
}

//Q operator*(Q const & a, Q const & b) { return Q(operator*((Q::Super)a,(Q::Super)b)); }
template<typename A, typename B> decltype(auto) operator+(quat<A> const & a, quat<B> const & b) { using S = decltype(A() + B()); return quat<S>(typename quat<A>::Super(a) + typename quat<B>::Super(b)); }
template<typename A, typename B> decltype(auto) operator-(quat<A> const & a, quat<B> const & b) { using S = decltype(A() - B()); return quat<S>(typename quat<A>::Super(a) - typename quat<B>::Super(b)); }
template<typename A, typename B> decltype(auto) operator/(quat<A> const & a, quat<B> const & b) { using S = decltype(A() / B()); return quat<S>(typename quat<A>::Super(a) / typename quat<B>::Super(b)); }
template<typename A, typename B> decltype(auto) operator+(quat<A> const & a, B const & b) { using S = decltype(A() + B()); return quat<S>(typename quat<A>::Super(a) + b); }
template<typename A, typename B> decltype(auto) operator-(quat<A> const & a, B const & b) { using S = decltype(A() - B()); return quat<S>(typename quat<A>::Super(a) - b); }
template<typename A, typename B> decltype(auto) operator*(quat<A> const & a, B const & b) { using S = decltype(A() * B()); return quat<S>(typename quat<A>::Super(a) * b); }
template<typename A, typename B> decltype(auto) operator/(quat<A> const & a, B const & b) { using S = decltype(A() / B()); return quat<S>(typename quat<A>::Super(a) / b); }
template<typename A, typename B> decltype(auto) operator+(A const & a, quat<B> const & b) { using S = decltype(A() + B()); return quat<S>(a + typename quat<B>::Super(b)); }
template<typename A, typename B> decltype(auto) operator-(A const & a, quat<B> const & b) { using S = decltype(A() - B()); return quat<S>(a - typename quat<B>::Super(b)); }
template<typename A, typename B> decltype(auto) operator*(A const & a, quat<B> const & b) { using S = decltype(A() * B()); return quat<S>(a * typename quat<B>::Super(b)); }
template<typename A, typename B> decltype(auto) operator/(A const & a, quat<B> const & b) { using S = decltype(A() / B()); return quat<S>(a / typename quat<B>::Super(b)); }

template<typename T>
quat<T> normalize(quat<T> const & q) {
	T len = q.length();
	if (len <= quat<T>::angleAxisEpsilon) return quat<T>();
	return (quat<T>)(q / len);
}

template<typename T>
std::ostream & operator<<(std::ostream & o, quat<T> const & q) {
	char const * seporig = "";
	char const * sep = seporig;
	for (int i = 0; i < 4; ++i) {
		auto const & qi = q[i];
		if (qi != 0) {
			o << sep;
			if (qi == -1) {
				o << "-";
			} else if (qi != 1) {
				o << qi << "*";
			}
			o << "e_" << ((i + 1) % 4);	// TODO quaternion indexing ...
			sep = " + ";
		}
	}
	if (sep == seporig) {
		return o << "0";
	}
	return o;
}



using quati = quat<int>;	// I don't judge
using quatf = quat<float>;
using quatd = quat<double>;

}
