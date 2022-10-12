#pragma once

#include <algorithm>

namespace Tensor {

template<int rank_>
struct RangeObj {
	static constexpr auto rank = rank_;
	using intN = Tensor::intN<rank>;
	intN min, max;

	RangeObj(intN min_, intN max_) : min(min_), max(max_) {}

	// iterators

	struct iterator {
		intN index, min, max;
		
		iterator()  {}
		iterator(intN min_, intN max_) : index(min_), min(min_), max(max_) {}
		iterator(iterator const & iter) : index(iter.index), min(iter.min), max(iter.max) {}
		iterator(iterator && iter) : index(iter.index), min(iter.min), max(iter.max) {}
		
		bool operator==(iterator const &b) const { return index == b.index; }
		bool operator!=(iterator const &b) const { return index != b.index; }

		//converts index to int
		int flatten() const {
			int flatIndex = 0;
			for (int i = rank - 1; i >= 0; --i) {
				flatIndex *= max(i) - min(i);
				flatIndex += index(i) - min(i);
			}
			return flatIndex;
		}
	
		//converts int to index
		void unflatten(int flatIndex) {
			for (int i = 0; i < rank; ++i) {
				int s = max(i) - min(i);
				int n = flatIndex;
				if (i < rank-1) n %= s;
				index(i) = n + min(i);
				flatIndex = (flatIndex - n) / s;
			}
		}

		iterator &operator+=(int offset) {
			unflatten(flatten() + offset);
			return *this;
		}

		iterator &operator-=(int offset) {
			unflatten(flatten() - offset);
			return *this;
		}

		iterator operator+(int offset) const {
			return iterator(*this) += offset;
		}

		iterator operator-(int offset) const {
			return iterator(*this) -= offset;
		}

		int operator-(iterator const &i) const {
			return flatten() - i.flatten();
		}

		iterator &operator++() {
			for (int i = 0; i < rank; ++i) {	//allow the last index to overflow for sake of comparing it to end
				++index(i);
				if (index(i) < max(i)) break;
				if (i < rank-1) index(i) = min(i);
			}
			return *this;
		}

		intN &operator*() { return index; }
		intN *operator->() { return &index; }
	
		static iterator end(intN min, intN max) {
			iterator i(min, max);
			i.index(rank-1) = i.max(rank-1);
			return i;
		}
	};

	iterator begin() { return iterator(min, max); }
	iterator end() { return iterator::end(min, max); }

	using const_iterator = iterator;
	const_iterator begin() const { return const_iterator(min, max); }
	const_iterator end() const { return iterator::end(min, max); }
};

}
