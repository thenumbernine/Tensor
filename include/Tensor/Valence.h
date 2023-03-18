#pragma once

/*
Valence is a thin wrapper around tensor<> 
except that when indexing it, it produces a valenceIndex thin wrapper around Index
*/

namespace Tensor {

template<typename T, typename ... Vs>
struct valence : public T {
	using Super = T;
	using Super::Super;

	// now forward all arguments (and their respective require conditions?  man ...)
	// *EXCEPT* for the Index operator one
	// for that, have it return a valence wrapper
};

}
