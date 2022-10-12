#pragma once

// wrapper of a ref of a type that reads and writes the negative of the value its wrapping

#include "Tensor/Meta.h"	//is_tensor_v, is_instance_v
#include <functional>	//std::reference_wrapper
#include <optional>
#include <ostream>		//std::ostream

namespace Tensor {

struct Sign {
	typedef enum {
		POSITIVE, // == 0 // so that how1 ^ how2 produces the sign for AntiSymRef1-of-AntiSymRef2
		NEGATIVE, // == 1
		ZERO,
	} Value;
	Value value = POSITIVE;
	Sign() {}
	Sign(Value const & value_) : value(value_) {}
	Sign(Value && value_) : value(value_) {}
	Sign & operator=(Value const & value_) {
		value = value_;
		return *this;
	}
	bool operator==(Value const & value_) const { return value == value_; }
	bool operator!=(Value const & value_) const { return !operator==(value_); }
};

inline std::ostream& operator<<(std::ostream & o, Sign const & s) {
	return o << "Sign(" << s.value << ")";
}

inline Sign operator*(Sign a, Sign b) {
	if ((a == Sign::POSITIVE || a == Sign::NEGATIVE)
		&& (b == Sign::POSITIVE || b == Sign::NEGATIVE))
	{
		return (Sign::Value)((int)a.value ^ (int)b.value);
	}
	return Sign::ZERO;
}

inline Sign operator!(Sign a) {
	return a * Sign::NEGATIVE;
}

template<typename T>
struct AntiSymRef {
	using This = AntiSymRef;
	using Type = T;
	
	static_assert(!is_instance_v<T, AntiSymRef>);	//never wrap AntiSymRef<AntiSymRef<T>>, instead just use 1 wrapper, flip 'how' if necessary.

	std::optional<std::reference_wrapper<T>> x;

	Sign how = Sign::ZERO;
	
	AntiSymRef() {}
	AntiSymRef(std::reference_wrapper<T> const & x_, Sign how_) : x(x_), how(how_) {}
	AntiSymRef(AntiSymRef const & r) : x(r.x), how(r.how) {}
	AntiSymRef(AntiSymRef && r) : x(r.x), how(r.how) {}
	
	This & operator=(T const & y) {
		if (how == Sign::POSITIVE) {
			(*x).get() = y;
		} else if (how == Sign::NEGATIVE) {
			(*x).get() = -y;
		} else {//if (how == ZERO) {
			//ZERO should only be wrapping temp elements outside the antisymmetric matrix
		}
		return *this;
	}
	
	This & operator=(T && y) {
		if (how == Sign::POSITIVE) {
			(*x).get() = y;
		} else if (how == Sign::NEGATIVE) {
			(*x).get() = -y;
		} else {//if (how == ZERO) {
			// no write
		}
		return *this;
	}

	operator T() const {
		if (how == Sign::POSITIVE) {
			return (*x).get();
		} else if (how == Sign::NEGATIVE) {
			return -(*x).get();
		} else {//if (how == ZERO) {
			return {};
		}
	}

	template<typename... Args>
	auto operator()(Args&&... args) const
	//requires (std::is_invocable_v<T()>) 
	requires (is_tensor_v<T>)
	{
		using R = std::decay_t<decltype((*x).get()(std::forward<Args>(args)...))>;
		if constexpr (is_instance_v<R, AntiSymRef>) {
			using RT = typename R::Type;	// TODO nested-most?  verified for two-nestings deep, what about 3?
			using RRT = std::conditional_t<std::is_const_v<T>, const RT, RT>;
			if (how != Sign::POSITIVE && how != Sign::NEGATIVE) {
				return AntiSymRef<RRT>();
			} else {
				R && r = (*x).get()(std::forward<Args>(args)...);
				if (r.how != Sign::POSITIVE && r.how != Sign::NEGATIVE) {
					return AntiSymRef<RRT>();
				} else {
					return AntiSymRef<RRT>(*r.x, how * r.how);
				}
			}
		} else {
			using RRT = std::conditional_t<std::is_const_v<T>, const R, R>;
			if (how != Sign::POSITIVE && how != Sign::NEGATIVE) {
				return AntiSymRef<RRT>();
			} else {
				return AntiSymRef<RRT>(std::ref(
					(*x).get()(std::forward<Args>(args)...)
				), how);
			}
		}
	}

	// return ref or no?
	constexpr This flip() const {
		if (how == Sign::POSITIVE || how == Sign::NEGATIVE) {
			return AntiSymRef(*x, !how);
		}
		return *this;
	}
};

static_assert(is_instance_v<AntiSymRef<double>, AntiSymRef>);
static_assert(is_instance_v<AntiSymRef<AntiSymRef<double>>, AntiSymRef>);

template<typename T>
bool operator==(AntiSymRef<T> const & a, T const & b) {
	return a.operator T() == b;
}

template<typename T>
bool operator==(T const & a, AntiSymRef<T> const & b) {
	return a == b.operator T();
}

template<typename T>
bool operator==(AntiSymRef<T> const & a, AntiSymRef<T> const & b) {
	return a.operator T() == b.operator T();
}

template<typename T> bool operator!=(AntiSymRef<T> const & a, T const & b) { return !operator==(a,b); }
template<typename T> bool operator!=(T const & a, AntiSymRef<T> const & b) { return !operator==(a,b); }
template<typename T> bool operator!=(AntiSymRef<T> const & a, AntiSymRef<T> const & b) { return !operator==(a,b); }

template<typename T>
std::ostream & operator<<(std::ostream & o, AntiSymRef<T> const & t) {
	return o << t.operator T();
}

}
