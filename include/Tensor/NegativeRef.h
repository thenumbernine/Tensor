#pragma once

// wrapper of a ref of a type that reads and writes the negative of the value its wrapping

#include <functional>	//std::reference_wrapper

namespace Tensor {

template<typename T>
struct NegativeRef {
	using This = NegativeRef;
	
	std::reference_wrapper<T> x;
	
	NegativeRef(std::reference_wrapper<T> const & x_) : x(x_) {}
	
	NegativeRef(std::reference_wrapper<T> && x_) : x(x_) {}
	
	This & operator=(T const & y) {
		x.get() = -y;
		return *this;
	}
	
	This & operator=(T && y) {
		x.get() = -y;
		return *this;
	}

	operator T() const { return -x; }
};

}
