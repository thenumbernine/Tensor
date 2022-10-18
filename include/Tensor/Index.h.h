#pragma once

#include "Tensor/Meta.h"	//is_tensor_v
#include <utility>			//integer_sequence
#include <tuple>			//tuple

namespace Tensor {

//index-access classes
struct IndexBase;

template<char ident>
struct Index;

//forward-declare for index-access
template<typename InputTensorType, typename IndexVector>
requires is_tensor_v<InputTensorType>
struct IndexAccess;

// Vector.h needs this:
template<
	typename IndexTuple,
	typename indexSeq = std::make_integer_sequence<int, std::tuple_size_v<IndexTuple>>
>
struct GatherIndexesImpl;
template<typename IndexTuple>
using GatherIndexes = typename GatherIndexesImpl<
	IndexTuple
>::type;

// used for picking out the summation indexes, used both by IndexAccess and by determining in tensors if the trace produces a scalar or not
template<typename T>
struct GetIthTupleSecond {
	template<int i>
	using go = typename std::tuple_element_t<i, T>::second_type;
};

// helper for GatherIndexes' results combined with tuple_get_filtered_indexes_t for picking out the IndexTuple locs from the results
template<typename Locs, typename GatheredIndexes>
using GetIndexLocsFromGatherResult = Common::tuple_apply_t<
	Common::seq_cat_t,
	Common::tuple_cat_t<	
		std::tuple<int>,	//Locs better be an int sequence
		Common::SeqToTupleMap<
			Locs,
			GetIthTupleSecond<GatheredIndexes>::template go
		>
	>
>;

// T is a member of the tuple within GatherIndexes<>'s results
// for use with GatheredIndexes's results
template<typename T>
struct HasMoreThanOneIndex {
	static_assert(T::second_type::size() == 1 || T::second_type::size() == 2, "indexes can only appear 1 or 2 times, no more than that.");
	static constexpr bool value = T::second_type::size() == 2;
};

}
