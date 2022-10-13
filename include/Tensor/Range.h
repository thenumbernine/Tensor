#pragma once

#include <algorithm>

namespace Tensor {
	
/*
inc index 0 first (and flattening adding order 0 last) is memory-in-order for row-major images (since the 1st coordinate is the column)
but inc index rank-1 first (and flattening adding it last) is not memory-in-order for row-major matrices (since the 1st coordinate is the row)
so for memory-in-order matrix iteration, use OuterOrderIterator
but for memory-in-order image iteration, use InnerOrderIterator
or ofc you can switch matrixes to be col-major, like GLSL, and then InnerOrderIterator works for both, but then A_ij in math = A.j.i in code
*/
template<int rank_, bool innerFirst>
struct RangeObj {
	static constexpr auto rank = rank_;
	using intN = Tensor::intN<rank>;
	intN min, max;

	constexpr RangeObj(intN min_, intN max_) : min(min_), max(max_) {}

	template<int rankFirst, int rankLast, int rankStep>
	struct IteratorOrder {
		RangeObj const & owner;
		intN index;
		
		IteratorOrder(RangeObj const & owner_) : owner(owner_), index(owner_.min) {}
		IteratorOrder(RangeObj const & owner_, intN index_) : owner(owner_), index(index_) {}
		IteratorOrder(IteratorOrder const & iter) : owner(iter.owner), index(iter.index) {}
		IteratorOrder(IteratorOrder && iter) : owner(iter.owner), index(iter.index) {}
		
		template<int i>
		struct Inc {
			static constexpr bool exec(IteratorOrder & it) {
				constexpr int j = rankFirst + i * rankStep;
				++it.index[j];
				if (it.index[j] < it.owner.max[j]) return true;
				if (j != rankLast) it.index[j] = it.owner.min[j];
				return false;
			}
		};
		
		//converts index to int
		constexpr int flatten() const {
			int flatIndex = 0;
			for (int i = rankLast; i != rankFirst-rankStep; i -= rankStep) {
				flatIndex *= owner.max[i] - owner.min[i];
				flatIndex += index[i] - owner.min[i];
			}
			return flatIndex;
		}
	
		//converts int to index
		constexpr void unflatten(int flatIndex) {
			for (int i = rankFirst; i != rankLast+rankStep; i += rankStep) {
				int s = owner.max[i] - owner.min[i];
				int n = flatIndex;
				if (i != rankLast) n %= s;
				index[i] = n + owner.min[i];
				flatIndex = (flatIndex - n) / s;
			}
		}
	
		bool operator==(IteratorOrder const & b) const { return index == b.index; }
		bool operator!=(IteratorOrder const & b) const { return index != b.index; }
		IteratorOrder & operator+=(int offset) { unflatten(flatten() + offset); return *this; }
		IteratorOrder & operator-=(int offset) { unflatten(flatten() - offset); return *this; }
		IteratorOrder operator+(int offset) const { return IteratorOrder(*this) += offset; }
		IteratorOrder operator-(int offset) const { return IteratorOrder(*this) -= offset; }
		int operator-(IteratorOrder const &i) const { return flatten() - i.flatten(); }
		IteratorOrder & operator++() { Common::ForLoop<0, rank, Inc>::exec(*this); return *this; }
		intN &operator*() { return index; }
		intN *operator->() { return &index; }
	
		static constexpr IteratorOrder end(RangeObj const & owner) {
			auto i = IteratorOrder(owner);
			i.index[rankLast] = owner.max[rankLast];
			return i;
		}
	};

	using InnerOrderIterator = IteratorOrder<0, rank-1, 1>;		// inc 0 first
	using OuterOrderIterator = IteratorOrder<rank-1, 0, -1>;	// inc n-1 first

	// iterators

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
