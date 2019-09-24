#pragma once

#include "Tensor/GenericVector.h"
#include "Tensor/Vector.h"

namespace Tensor {

template<int dim_>
struct GenericRank1 {
	static constexpr auto dim = dim_;
	static constexpr auto rank = 1;

	//now i need a specialization of this for when 'type' is a primitive
	// in such cases, accessing nested members results in compiler errors
	template<typename InnerType, typename ScalarType>
	struct Body : public GenericVector<InnerType, dim, ScalarType, Body<InnerType, ScalarType>> {
		using Parent = GenericVector<InnerType, dim, ScalarType, Body<InnerType, ScalarType>>;
		
		using Parent::Parent;

		InnerType &operator()(const Vector<int,1> &deref) { return Parent::v[deref(0)]; }
		const InnerType &operator()(const Vector<int,1> &deref) const { return Parent::v[deref(0)]; }
	
		//formality for decoding write indexes
		static Vector<int,1> getReadIndexForWriteIndex(int writeIndex) { return Vector<int,1>(writeIndex); }
	};
};

}
