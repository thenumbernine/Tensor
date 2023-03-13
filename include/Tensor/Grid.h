#pragma once

#include "Tensor/Vector.h"	// new tensor struct
#include "Tensor/Range.h"
#include "Common/Exception.h"
#include <cassert>

#if PLATFORM_MSVC
#undef min
#undef max
#endif


namespace Tensor {

//rank is templated, but dim is not as it varies per-rank
//so this is dynamically-sized tensor
template<typename Type_, int rank_>
struct Grid {
	using Type = Type_;
	using value_type = Type;
	static constexpr auto rank = rank_;
	using intN = Tensor::intN<rank>;

	intN size;
	Type * v = {};
	bool own = {};	//or just use a shared_ptr to v?
	
	//cached for quick access by dot with index vector
	//step[0] = 1, step[1] = size[0], step[j] = product(i=1,j-1) size[i]
	intN step;

	Grid(intN const & size_ = intN(), Type* v_ = {})
	:	size(size_),
		v(v_)
	{
		if (!v) {
			v = new Type[size.product()]();
			own = true;
		}
		step(0) = 1;
		for (int i = 1; i < rank; ++i) {
			step(i) = step(i-1) * size(i-1);
		}
	}

	Grid(intN const & size_, std::function<Type(intN)> f)
	:	size(size_)
	{
		v = new Type[size.product()]();
		own = true;
		
		step(0) = 1;
		for (int i = 1; i < rank; ++i) {
			step(i) = step(i-1) * size(i-1);
		}
	
		for (auto i : range()) {
			(*this)(i) = f(i);
		}
	}


	~Grid() {
		if (own) {
			delete[] v;
		}
	}

	// dereference by vararg ints

	template<int offset, typename... Rest>
	struct BuildDeref;

	template<int offset, typename... Rest>
	struct BuildDeref<offset, int, Rest...> {
		static intN exec(int next, Rest ... rest) {
			intN index = BuildDeref<offset+1, Rest...>::exec(rest...);
			index(offset) = next;
			return index;
		}
	};

	template<int offset>
	struct BuildDeref<offset, int> {
		static intN exec(int last) {
			static_assert(offset == rank-1, "didn't provide enough arguments for dereference");
			intN index;
			index(offset) = last;
			return index;
		}
	};

	template<typename... Rest>
	Type &operator()(int first, Rest... rest) {
		return getValue(BuildDeref<0, int, Rest...>::exec(first, rest...));
	}

	template<typename... Rest>
	Type const &operator()(int first, Rest... rest) const {
		return getValue(BuildDeref<0, int, Rest...>::exec(first, rest...));
	}

	//dereference by a vector of ints

	//typical access will be only for the Type's sake
	Type &operator()(intN const &deref) { return getValue(deref); }
	Type const &operator()(intN const &deref) const { return getValue(deref); }

	//but other folks (currently only our initialization of our indexes) will want the whole value
	Type &getValue(intN const &deref) { 
#ifdef DEBUG
		for (int i = 0; i < rank; ++i) {
			if (deref(i) < 0 || deref(i) >= size(i)) {
				throw Common::Exception() << "size is " << size << " but dereference is " << deref;
			}
		}
#endif
		int flat_deref = deref.dot(step);
		assert(flat_deref >= 0 && flat_deref < size.product());
		return v[flat_deref];
	}
	Type const &getValue(intN const &deref) const { 
#ifdef DEBUG
		for (int i = 0; i < rank; ++i) {
			if (deref(i) < 0 || deref(i) >= size(i)) {
				throw Common::Exception() << "size is " << size << " but dereference is " << deref;
			}
		}
#endif
		int flat_deref = deref.dot(step);
		assert(flat_deref >= 0 && flat_deref < size.product());
		return v[flat_deref];
	}

	using iterator = Type*;
	using const_iterator = Type const*;
	Type *begin() { return v; }
	Type *end() { return v + size.product(); }
	Type const *begin() const { return v; }
	Type const *end() const { return v + size.product(); }

	RangeObj<rank> range() const {
		return RangeObj<rank>(intN(), size);
	}

	//dereference by vararg ints 
	
	template<typename... Rest>
	void resize(int first, Rest... rest) {
		resize(BuildDeref<0, int, Rest...>::exec(first, rest...));
	}

	//dereference by a vector of ints

	void resize(intN const& newSize) {
		if (size == newSize) return;
		
		intN oldSize = size;
		intN oldStep = step;
		Type* oldV = v;
		
		size = newSize;
		v = new Type[newSize.product()];
		step(0) = 1;
		for (int i = 1; i < rank; ++i) {
			step(i) = step(i-1) * size(i-1);
		}

		intN minSize;
		for (int i = 0; i < rank; ++i) {
			minSize(i) = size(i) < oldSize(i) ? size(i) : oldSize(i);
		}

		RangeObj<rank> range(intN(), minSize);		
		for (typename RangeObj<rank>::iterator iter = range.begin(); iter != range.end(); ++iter) {
			intN index = *iter;
			int oldOffset = oldStep.dot(index);
			(*this)(index) = oldV[oldOffset];
		}

		delete[] oldV;
	}

	Grid& operator=(Grid& src) {
		resize(src.dims());

		Type* srcv = src.v;
		Type* dstv = v;
		for (int i = size.product()-1; i >= 0; --i) {
			*(dstv++) = *(srcv++);
		}
		return *this;
	}
};

}
