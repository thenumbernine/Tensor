#pragma once

namespace Tensor {

template<
	int rankFirst,
	int rankLast,
	int rankStep,
	int rank,
	typename Owner
>
struct RangeIterator;
// this one is in Tensor/Range.h

template<
	int rank, 
	typename Owner
> using RangeIteratorInner = RangeIterator<0, rank-1, 1,  rank, Owner>;

template<
	int rank, 
	typename Owner
> using RangeIteratorOuter = RangeIterator<rank-1, 0, -1, rank, Owner>;

template<
	int rank, 
	bool innerFirst,
	typename Owner
> using RangeIteratorInnerVsOuter = std::conditional_t<
	innerFirst,
	RangeIteratorInner<rank, Owner>,
	RangeIteratorOuter<rank, Owner>
>;

template<int rank_, bool innerFirst = true>
struct RangeObj;

}
