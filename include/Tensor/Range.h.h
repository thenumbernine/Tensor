#pragma once

namespace Tensor {

template<
	int rankFirst,
	int rankLast,
	int rankStep,
	int rank,
	typename Owner,
	int getMin(Owner const &, int),
	int getMax(Owner const &, int)
>
struct RangeIterator;
// this one is in Tensor/Range.h

template<
	int rank, 
	bool innerFirst,
	typename Owner,
	int getMin(Owner const &, int),
	int getMax(Owner const &, int)
> using RangeIteratorInnerVsOuter = std::conditional_t<
	innerFirst,
	RangeIterator<0, rank-1, 1,  rank, Owner, getMin, getMax>,
	RangeIterator<rank-1, 0, -1, rank, Owner, getMin, getMax>
>;

template<int rank_, bool innerFirst = true>
struct RangeObj;

}
