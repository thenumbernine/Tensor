#pragma once

#include "TensorMath/GenericVector.h"
#include "TensorMath/Vector.h"

template<int dim_>
struct GenericRank1 {
	enum { dim = dim_ };
	enum { rank = 1 };

	//now i need a specialization of this for when 'type' is a primitive
	// in such cases, accessing nested members results in compiler errors
	template<typename InnerType, typename ScalarType>
	struct Body : public GenericVector<InnerType, dim, ScalarType, Body<InnerType, ScalarType>> {
		typedef GenericVector<InnerType, dim, ScalarType, Body<InnerType, ScalarType>> Parent;
		
		using Parent::Parent;

		InnerType &operator()(const Vector<int,1> &deref) { return Parent::v[deref(0)]; }
		const InnerType &operator()(const Vector<int,1> &deref) const { return Parent::v[deref(0)]; }
	
		//formality for decoding write indexes
		static Vector<int,1> getReadIndexForWriteIndex(int writeIndex) { return Vector<int,1>(writeIndex); }
	};
};

