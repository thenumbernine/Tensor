#pragma once

#include "TensorMath/Vector.h"
#include <assert.h>
#include <algorithm>

//TODO move RangeObj iterator with Grid iterator
template<int rank_>
struct RangeObj {
	enum { rank = rank_ };
	typedef Vector<int,rank> DerefType;
	DerefType min, max;

	RangeObj(DerefType min_, DerefType max_) : min(min_), max(max_) {}

	// iterators

	struct iterator {
		DerefType index, min, max;
		
		iterator()  {}
		iterator(DerefType min_, DerefType max_) : index(min_), min(min_), max(max_) {}
		iterator(const iterator &iter) : index(iter.index), min(iter.min), max(iter.max) {}
		
		bool operator==(const iterator &b) const { return index == b.index; }
		bool operator!=(const iterator &b) const { return index != b.index; }
	
		int flatten() {
			int flatIndex = 0;
			for (int i = rank - 1; i <= 0; --i) {
				flatIndex *= max(i) - min(i);
				flatIndex += index(i) - min(i);
			}
			return flatIndex;
		}
		
		int operator-(const iterator &i) {
			return flatten() - i.flatten();
		}

		//poor man's way to do this

		iterator &operator++() {
			for (int i = 0; i < rank; ++i) {	//allow the last index to overflow for sake of comparing it to end
				++index(i);
				if (index(i) < max(i)) break;
				if (i < rank-1) index(i) = min(i);
			}
			return *this;
		}

		DerefType &operator*() { return index; }
		DerefType *operator->() { return &index; }
	};

	iterator begin() {
		return iterator(min, max);
	}
	iterator end() {
		iterator i(min, max);
		i.index(rank-1) = i.max(rank-1);
		return i;
	}
	
	struct const_iterator {
		DerefType index, min, max;
		
		const_iterator() {}
		const_iterator(DerefType min_, DerefType max_) : min(min_), max(max_) {}
		const_iterator(const const_iterator &iter) : index(iter.index), min(iter.min), max(iter.max) {}
		
		bool operator==(const const_iterator &b) const { return index == b.index; }
		bool operator!=(const const_iterator &b) const { return index != b.index; }
		
		const_iterator &operator++() {
			for (int i = 0; i < rank; ++i) {	//allow the last index to overflow for sake of comparing it to end
				++index(i);
				if (index(i) < max(i)) break;
				if (i < rank-1) index(i) = min(i);
			}
			return *this;
		}

		const DerefType &operator*() const { return index; }
		const DerefType *operator->() const { return &index; }
	};

	const_iterator begin() const {
		return const_iterator(min, max);
	}
	const_iterator end() const {
		const_iterator i(min, max);
		i.index(rank-1) = i.max(rank-1);
		return i;
	}
};


//rank is templated, but dim is not as it varies per-rank
//so this is dynamically-sized tensor
template<typename Type_, int rank_>
struct Grid {
	typedef Type_ Type;
	typedef Type value_type;
	enum { rank = rank_ };
	typedef Vector<int,rank> DerefType;

	Type *v;
	DerefType size;
	
	//cached for quick access by dot with index vector
	//step[0] = 1, step[1] = size[0], step[j] = product(i=1,j-1) size[i]
	DerefType step;

	Grid(const DerefType &size_) : size(size_) {
		v = new Type[size.volume()]();
		step(0) = 1;
		for (int i = 1; i < rank; ++i) {
			step(i) = step(i-1) * size(i-1);
		}
	}

	~Grid() {
		delete[] v;
	}

	//typical access will be only for the Type's sake
	Type &operator()(const DerefType &deref) { return getValue(deref); }
	const Type &operator()(const DerefType &deref) const { return getValue(deref); }

	//but other folks (currently only our initialization of our indexes) will want the whole value
	Type &getValue(const DerefType &deref) { 
#ifdef DEBUG
		for (int i = 0; i < rank; ++i) {
			if (deref(i) < 0 || deref(i) >= size(i)) {
				//why aren't callstacks recorded when exceptions are thrown?
				std::cerr << "size is " << size << " but dereference is " << deref << std::endl;
				assert(false);
			}
		}
#endif
		int flat_deref = DerefType::dot(deref, step);
		assert(flat_deref >= 0 && flat_deref < size.volume());
		return v[flat_deref];
	}
	const Type &getValue(const DerefType &deref) const { 
#ifdef DEBUG
		for (int i = 0; i < rank; ++i) {
			if (deref(i) < 0 || deref(i) >= size(i)) {
				std::cerr << "size is " << size << " but dereference is " << deref << std::endl;
				assert(false);
			}
		}
#endif
		int flat_deref = DerefType::dot(deref, step);
		assert(flat_deref >= 0 && flat_deref < size.volume());
		return v[flat_deref];
	}

	Type *begin() { return v; }
	Type *end() { return v + size.volume(); }
	const Type *begin() const { return v; }
	const Type *end() const { return v + size.volume(); }
};

