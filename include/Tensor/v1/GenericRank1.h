#pragma once

#include "Tensor/v1/GenericVector.h"
#include "Tensor/v1/Vector.h"

namespace Tensor {
namespace v1 {

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

		InnerType &operator()(int1 const &deref) { return Parent::v[deref(0)]; }
		InnerType const &operator()(int1 const &deref) const { return Parent::v[deref(0)]; }
		
		InnerType &operator[](int i) { return Parent::v[i]; }
		InnerType const &operator[](int i) const { return Parent::v[i]; }
	
		//formality for decoding write indexes
		static int1 getReadIndexForWriteIndex(int writeIndex) { return int1(writeIndex); }
	};
};

}
}
