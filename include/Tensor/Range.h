#pragma once

#include "Tensor/Range.h.h"
#include <algorithm>

namespace Tensor {

/*
inc index 0 first (and flattening adding order 0 last) is memory-in-order for row-major images (since the 1st coordinate is the column)
but inc index rank-1 first (and flattening adding it last) is not memory-in-order for row-major matrices (since the 1st coordinate is the row)
so for memory-in-order matrix iteration, use OuterOrderIterator
but for memory-in-order image iteration, use InnerOrderIterator
or ofc you can switch matrixes to be col-major, like GLSL, and then InnerOrderIterator works for both, but then A_ij in math = A.j.i in code
*/
template<
	int rankFirst,
	int rankLast,
	int rankStep,
	int rank,
	typename Owner,
	int getMin(Owner const &, int),
	int getMax(Owner const &, int)
>
struct RangeIterator {
	Owner const & owner;
	using intN = Tensor::intN<rank>;
	intN index;
	
	RangeIterator(Owner const & owner_) : owner(owner_), index([&](int j) -> int { return getMin(owner,j); }) {}
	RangeIterator(Owner const & owner_, intN index_) : owner(owner_), index(index_) {}
	RangeIterator(RangeIterator const & iter) : owner(iter.owner), index(iter.index) {}
	RangeIterator(RangeIterator && iter) : owner(iter.owner), index(iter.index) {}
	
	template<int i>
	struct Inc {
		static constexpr bool exec(RangeIterator & it) {
			constexpr int j = rankFirst + i * rankStep;
			++it.index[j];
			if (it.index[j] < getMax(it.owner, j)) return true;
			if (j != rankLast) it.index[j] = getMin(it.owner, j);
			return false;
		}
	};
	
	//converts index to int
	constexpr int flatten() const {
		int flatIndex = 0;
		for (int i = rankLast; i != rankFirst-rankStep; i -= rankStep) {
			flatIndex *= getMax(owner, i) - getMin(owner, i);
			flatIndex += index[i] - getMin(owner, i);
		}
		return flatIndex;
	}

	//converts int to index
	constexpr void unflatten(int flatIndex) {
		for (int i = rankFirst; i != rankLast+rankStep; i += rankStep) {
			int s = getMax(owner, i) - getMin(owner, i);
			int n = flatIndex;
			if (i != rankLast) n %= s;
			index[i] = n + getMin(owner, i);
			flatIndex = (flatIndex - n) / s;
		}
	}

	bool operator==(RangeIterator const & b) const { return index == b.index; }
	bool operator!=(RangeIterator const & b) const { return index != b.index; }
	RangeIterator & operator+=(int offset) { unflatten(flatten() + offset); return *this; }
	RangeIterator & operator-=(int offset) { unflatten(flatten() - offset); return *this; }
	RangeIterator operator+(int offset) const { return RangeIterator(*this) += offset; }
	RangeIterator operator-(int offset) const { return RangeIterator(*this) -= offset; }
	int operator-(RangeIterator const &i) const { return flatten() - i.flatten(); }
	RangeIterator & operator++() { Common::ForLoop<0, rank, Inc>::exec(*this); return *this; }
	intN &operator*() { return index; }
	intN *operator->() { return &index; }

	static constexpr RangeIterator end(Owner const & owner) {
		auto i = RangeIterator(owner);
		i.index[rankLast] = getMax(owner, rankLast);
		return i;
	}
};

// iterate over an arbitrary range
template<int rank_, bool innerFirst>
struct RangeObj {
	using This = RangeObj;
	static constexpr auto rank = rank_;
	using intN = Tensor::intN<rank>;
	intN min, max;

	constexpr RangeObj(intN min_, intN max_) : min(min_), max(max_) {}

	static int getMin(This const & r, int i) { return r.min[i]; }
	static int getMax(This const & r, int i) { return r.max[i]; }
	using InnerOrderIterator = RangeIteratorInnerVsOuter<rank, true,  RangeObj, getMin, getMax>;	// inc 0 first
	using OuterOrderIterator = RangeIteratorInnerVsOuter<rank, false, RangeObj, getMin, getMax>;	// inc n-1 first

	using iterator = std::conditional_t<innerFirst, InnerOrderIterator, OuterOrderIterator>;
	iterator begin() { return iterator(*this); }
	iterator end() { return iterator::end(*this); }
	
	using const_iterator = iterator;
	const_iterator begin() const { return const_iterator(*this); }
	const_iterator end() const { return iterator::end(*this); }
	const_iterator cbegin() const { return const_iterator(*this); }
	const_iterator cend() const { return iterator::end(*this); }
};

}
