#pragma once

#include "TensorMath/Vector.h"
#include <assert.h>

//rank is templated, but dim is not as it varies per-rank
//so this is dynamically-sized tensor
template<typename Type_, int rank_>
struct Grid {
	typedef Type_ Type;
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

	Type &operator()(const DerefType &deref) { 
		for (int i = 0; i < rank; ++i) {
			assert(deref(i) >= 0 && deref(i) < size(i));
		}
		int flat_deref = DerefType::dot(deref, step);
		assert(flat_deref >= 0 && flat_deref < size.volume());
		return v[flat_deref];
	}
	const Type &operator()(const DerefType &deref) const { 
		for (int i = 0; i < rank; ++i) {
			assert(deref(i) >= 0 && deref(i) < size(i));
		}
		int flat_deref = DerefType::dot(deref, step);
		assert(flat_deref >= 0 && flat_deref < size.volume());
		return v[flat_deref];
	}

	//offset + clamp
	// because I wanted to write it concisely
	//but didn't want to offset by an entire vector, then clamp an entire vector
	Type &getOffset(DerefType deref, int dim, int offset) {
		deref(dim) = clamp(deref(dim) + offset, 0, size(dim)-1);
		return (*this)(deref);
	}
	const Type &getOffset(DerefType deref, int dim, int offset) const {
		deref(dim) = clamp(deref(dim) + offset, 0, size(dim)-1);
		return (*this)(deref);
	}


	// iterators

	struct iterator {
		Grid *parent;
		DerefType index;
		
		iterator() : parent(NULL) {}
		iterator(Grid *parent_) : parent(parent_) {}
		iterator(const iterator &iter) : parent(iter.parent), index(iter.index) {}
		
		bool operator==(const iterator &b) const { return index == b.index; }
		bool operator!=(const iterator &b) const { return index != b.index; }
		
		iterator &operator++() {
			for (int i = 0; i < rank; ++i) {	//allow the last index to overflow for sake of comparing it to end
				++index(i);
				if (index(i) < parent->size(i)) break;
				if (i < rank-1) index(i) = 0;
			}
			return *this;
		}

		typename Grid::Type &operator*() const { return (*parent)(index); }
		typename Grid::Type *operator->() const { return &((*parent)(index)); }
	};

	iterator begin() {
		iterator i(this);
		return i;
	}
	iterator end() {
		iterator i(this);
		i.index(rank-1) = size(rank-1);
		return i;
	}
	
	struct const_iterator {
		const Grid *parent;
		DerefType index;
		
		const_iterator() : parent(NULL) {}
		const_iterator(const Grid *parent_) : parent(parent_) {}
		const_iterator(const const_iterator &iter) : parent(iter.parent), index(iter.index) {}
		
		bool operator==(const const_iterator &b) const { return index == b.index; }
		bool operator!=(const const_iterator &b) const { return index != b.index; }
		
		const_iterator &operator++() {
			for (int i = 0; i < rank; ++i) {	//allow the last index to overflow for sake of comparing it to end
				++index(i);
				if (index(i) < parent->size(i)) break;
				if (i < rank-1) index(i) = 0;
			}
			return *this;
		}

		const typename Grid::Type &operator*() const { return (*parent)(index); }
		const typename Grid::Type *operator->() const { return &((*parent)(index)); }
	};

	const_iterator begin() const {
		const_iterator i(this);
		return i;
	}
	const_iterator end() const {
		const_iterator i(this);
		i.index(rank-1) = size(rank-1);
		return i;
	}
};

