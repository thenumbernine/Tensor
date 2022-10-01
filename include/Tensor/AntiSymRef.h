#pragma once

// wrapper of a ref of a type that reads and writes the negative of the value its wrapping

#include <functional>	//std::reference_wrapper
#include <optional>

namespace Tensor {

template<typename T>
struct AntiSymRef {
	using This = AntiSymRef;
	
	std::optional<std::reference_wrapper<T>> x;
	
	typedef enum {
		ZERO,
		POSITIVE,
		NEGATIVE,
	} How;
	How how = ZERO;
	
	AntiSymRef() {}
	AntiSymRef(std::reference_wrapper<T> const & x_, How how_) : x(x_), how(how_) {}
	AntiSymRef(AntiSymRef const & r) : x(r.x), how(x.how) {}
	
	AntiSymRef(AntiSymRef && r)
	: x(r.x), how(x.how) {}
	
	This & operator=(T const & y) {
		if (how == POSITIVE) {
			(*x).get() = y;
		} else if (how == NEGATIVE) {
			(*x).get() = -y;
		} else {//if (how == ZERO) {
			//ZERO should only be wrapping temp elements outside the antisymmetric matrix
		}
		return *this;
	}
	
	This & operator=(T && y) {
		if (how == POSITIVE) {
			(*x).get() = y;
		} else if (how == NEGATIVE) {
			(*x).get() = -y;
		} else {//if (how == ZERO) {
			// no write
		}
		return *this;
	}

	operator T() const {
		if (how == POSITIVE) {
			return *x;
		} else if (how == NEGATIVE) {
			return -*x;
		} else {//if (how == ZERO) {
			return {};
		}
	}
};

}
