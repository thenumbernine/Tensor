#pragma once

#include "Tensor/Range.h.h"
#include <algorithm>
#include <iostream>

namespace Tensor {

/*
inc index 0 first (and flattening adding order 0 last) is memory-in-order for row-major images (since the 1st coordinate is the column)
but inc index rank-1 first (and flattening adding it last) is not memory-in-order for row-major matrices (since the 1st coordinate is the row)
so for memory-in-order matrix iteration, use OuterOrderIterator
but for memory-in-order image iteration, use InnerOrderIterator
or ofc you can switch matrixes to be col-major, like GLSL, and then InnerOrderIterator works for both, but then A_ij in math = A.j.i in code

TODO con... template<int> int Owner::getRangeMin() and getRangeMax
*/
template<
	int rankFirst,
	int rankLast,
	int rankStep,
	int rank,
	typename Owner // should include the constness
>
struct RangeIterator {
	Owner & owner;
	using intN = Tensor::intN<rank>;
	intN index;

	template<int i>
	struct FillIndexWithMin {
		static constexpr bool exec(RangeIterator & it) {
			it.index[i] = it.owner.template getRangeMin<i>();
			return false;
		}
	};
	constexpr RangeIterator(Owner & owner_) : owner(owner_) {
		Common::ForLoop<0, rank-1, FillIndexWithMin>::exec(*this);
	}

	constexpr RangeIterator(Owner & owner_, intN index_) : owner(owner_), index(index_) {}
	constexpr RangeIterator(RangeIterator const & iter) : owner(iter.owner), index(iter.index) {}
	constexpr RangeIterator(RangeIterator && iter) : owner(iter.owner), index(iter.index) {}

	RangeIterator & operator=(RangeIterator const & o) {
		owner = o.owner;
		index = o.index;
		return *this;
	}
	RangeIterator & operator=(RangeIterator && o) {
		owner = o.owner;
		index = o.index;
		return *this;
	}

	template<int i>
	struct FlattenLoop {
		static constexpr bool exec(RangeIterator const & it, int & flatIndex) {
			flatIndex *= it.owner.template getRangeMax<i>() - it.owner.template getRangeMin<i>();
			flatIndex += it.index[i] - it.owner.template getRangeMin<i>();
			return false;
		}
	};
	//converts index to int
	constexpr int flatten() const {
		int flatIndex = 0;
		Common::ForLoop<rankLast, rankFirst-rankStep, FlattenLoop>::exec(*this, flatIndex);
		return flatIndex;
	}

	template<int i>
	struct UnFlattenLoop {
		static constexpr bool exec(RangeIterator & it, int & flatIndex) {
			int s = it.owner.template getRangeMax<i>() - it.owner.template getRangeMin<i>();
			int n = flatIndex;
			if (i != rankLast) n %= s;
			it.index[i] = n + it.owner.template getRangeMin<i>();
			flatIndex = (flatIndex - n) / s;
			return false;
		}
	};
	//converts int to index
	constexpr void unflatten(int flatIndex) {
		Common::ForLoop<rankFirst, rankLast+rankStep, UnFlattenLoop>::exec(*this, flatIndex);
	}

	constexpr bool operator==(RangeIterator const & b) const { return index == b.index; }
	constexpr bool operator!=(RangeIterator const & b) const { return index != b.index; }
	constexpr RangeIterator & operator+=(int offset) { unflatten(flatten() + offset); return *this; }
	constexpr RangeIterator & operator-=(int offset) { unflatten(flatten() - offset); return *this; }
	constexpr RangeIterator operator+(int offset) const { return RangeIterator(*this) += offset; }
	constexpr RangeIterator operator-(int offset) const { return RangeIterator(*this) -= offset; }
	constexpr int operator-(RangeIterator const &i) const { return flatten() - i.flatten(); }
	
	template<int i>
	struct Inc {
		static constexpr bool exec(RangeIterator & it) {
			++it.index[i];
			if (it.index[i] < it.owner.template getRangeMax<i>()) return true;
			if (i != rankLast) it.index[i] = it.owner.template getRangeMin<i>();
			return false;
		}
	};
	constexpr void inc() {
#if 1	// works but risks runaway template compiling
		Common::ForLoop<rankFirst, rankLast+rankStep, Inc>::exec(*this);
#elif 0	// works but still requires a helper class in global namespace ... still haven't got that constexpr lambda for template arg like I want ...
		Common::for_seq<
			Common::make_integer_range<int, rankFirst, rankLast+rankStep>,
			Inc
		>(*this);
#elif 0
		Common::static_foreach_seq(
			[&](int i) constexpr {
				Inc<i>::exec(*this);	// error: non-type template argument is not a constant expression 
			},
			Common::make_integer_range<int, rankFirst, rankLast+rankStep>()
		);
#endif
	}
	constexpr RangeIterator & operator++() { inc(); return *this; }
	constexpr RangeIterator & operator++(int) { inc(); return *this; }
	
	constexpr decltype(auto) operator*() const { return owner.getIterValue(index); }
	constexpr decltype(auto) operator->() const { return &owner.getIterValue(index); }

	static constexpr RangeIterator begin(Owner & owner) {
		return RangeIterator(owner);
	}

	static constexpr RangeIterator end(Owner & owner) {
		auto i = RangeIterator(owner);
		i.index[rankLast] = owner.template getRangeMax<rankLast>();
		return i;
	}

	// weird that this works with Common::has_to_ostream_v operator<< ... since nothing else does
	std::ostream & to_ostream(std::ostream & o) const {
		return o << "Iterator(owner=" << &owner << ", index=" << index << ")";
	}
};

// TODO intermediate class with templated getMin and getMax or something
// so I can use RangeObj iteration functionality but without having to store the min and max?

// iterate over an arbitrary range
template<int rank_, bool innerFirst>
struct RangeObj {
	using This = RangeObj;
	static constexpr auto rank = rank_;
	using intN = Tensor::intN<rank>;
	intN min, max;

	constexpr RangeObj(intN min_, intN max_) : min(min_), max(max_) {}

	// implementation for RangeIterator's Owner: 
	template<int i> constexpr int getRangeMin() const { return min[i]; }
	template<int i> constexpr int getRangeMax() const { return max[i]; }
	intN const & getIterValue(intN const & i) const { return i; }
	//

	using InnerOrderIterator = RangeIteratorInner<rank, RangeObj const>;	// inc 0 first
	using OuterOrderIterator = RangeIteratorOuter<rank, RangeObj const>;	// inc n-1 first

	using iterator = std::conditional_t<innerFirst, InnerOrderIterator, OuterOrderIterator>;
	constexpr iterator begin() { return iterator::begin(*this); }
	constexpr iterator end() { return iterator::end(*this); }
	
	using const_iterator = iterator;
	constexpr const_iterator begin() const { return const_iterator::begin(*this); }
	constexpr const_iterator end() const { return const_iterator::end(*this); }
	constexpr const_iterator cbegin() const { return const_iterator::begin(*this); }
	constexpr const_iterator cend() const { return const_iterator::end(*this); }
};

}
