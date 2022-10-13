#pragma once

#include <algorithm>

namespace Tensor {

template<int rank_, bool innerFirst>
struct RangeObj {
	static constexpr auto rank = rank_;
	using intN = Tensor::intN<rank>;
	intN min, max;

	constexpr RangeObj(intN min_, intN max_) : min(min_), max(max_) {}

	template<int rankFirst, int rankLast, int rankStep>
	struct IteratorOrder {
		
		intN index, min, max;
		
		IteratorOrder()  {}
		IteratorOrder(intN min_, intN max_) : index(min_), min(min_), max(max_) {}
		IteratorOrder(intN min_, intN max_, intN index_) : index(index_), min(min_), max(max_) {}
		IteratorOrder(IteratorOrder const & iter) : index(iter.index), min(iter.min), max(iter.max) {}
		IteratorOrder(IteratorOrder && iter) : index(iter.index), min(iter.min), max(iter.max) {}
		
		template<int i>
		struct Inc {
			static constexpr bool exec(IteratorOrder & it) {
				constexpr int j = rankFirst + i * rankStep;
				++it.index[j];
				if (it.index[j] < it.max[j]) return true;
				if (j != rankLast) it.index[j] = it.min[j];
				return false;
			}
		};
		
		//converts index to int
		constexpr int flatten() const {
			int flatIndex = 0;
			for (int i = rankLast; i != rankFirst-rankStep; i -= rankStep) {
				flatIndex *= max[i] - min[i];
				flatIndex += index[i] - min[i];
			}
			return flatIndex;
		}
	
		//converts int to index
		constexpr void unflatten(int flatIndex) {
			for (int i = rankFirst; i != rankLast+rankStep; i += rankStep) {
				int s = max[i] - min[i];
				int n = flatIndex;
				if (i != rankLast) n %= s;
				index[i] = n + min[i];
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
	
		static constexpr IteratorOrder end(intN min, intN max) {
			auto i = IteratorOrder(min, max);
			i.index[rankLast] = max[rankLast];
			return i;
		}
	};

	using InnerOrderIterator = IteratorOrder<0, rank-1, 1>;		// inc 0 first
	using OuterOrderIterator = IteratorOrder<rank-1, 0, -1>;	// inc n-1 first

	// iterators

	/*
	inc index 0 first (and flattening adding order 0 last) is memory-in-order for row-major images (since the 1st coordinate is the column)
	but inc index rank-1 first (and flattening adding it last) is not memory-in-order for row-major matrices (since the 1st coordinate is the row)
	so for memory-in-order matrix iteration, use OuterOrderIterator
	but for memory-in-order image iteration, use InnerOrderIterator
	or ofc you can switch matrixes to be col-major, like GLSL, and then InnerOrderIterator works for both, but then A_ij in math = A.j.i in code
	*/
	using iterator = std::conditional_t<innerFirst, InnerOrderIterator, OuterOrderIterator>;
		
	iterator begin() { return iterator(min, max); }
	iterator end() { return iterator::end(min, max); }

	using const_iterator = iterator;
	const_iterator begin() const { return const_iterator(min, max); }
	const_iterator end() const { return iterator::end(min, max); }
};

}
