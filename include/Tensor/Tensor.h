#pragma once

#include "Tensor/IndexPlace.h"	//not because tensor.h needs it, but because anyone using tensor.h needs it
#include "Tensor/Index.h"
#include "Common/Meta.h"
#include <functional>

namespace Tensor {

/*
new experimental template-driven tensor class
made to unify all of vector, one-form, symmetric, antisymmetric, and just regular kind matrixes
and any subsequently nested numeric structures.
*/

/*
support class for metaprogram specializations of tensor

rank 		= total rank of the structure. not necessarily the depth since some indexes (symmetric) can have >1 rank.
InnerType 	= the next-inner-most TensorStats type
BodyType 	= the BodyType (the generic_vector subclass) associated with this nesting level
*/
template<typename ScalarType, typename... Args>
struct TensorStats;

template<typename ScalarType_, typename Index_, typename... Args>
struct TensorStats<ScalarType_, Index_, Args...> {
	using ScalarType = ScalarType_;
	using Index = Index_;
	using InnerType = TensorStats<ScalarType, Args...>;
	
	//rank of the tensor.
	//vector indexes contribute 1's, matrix (sym & antisym) contribute 2, etc
	static constexpr auto rank = Index::rank + InnerType::rank;
	
	using BodyType = typename Index::template Body<typename InnerType::BodyType, ScalarType>;
	
		//inductive cases
	
	//get an element in a tensor
	template<int totalRank, int offset>
	static ScalarType &get(BodyType &body, intN<totalRank> const &deref) {
		intN<Index::rank> subderef;
		for (int i = 0; i < Index::rank; ++i) {
			subderef(i) = deref(i + offset);
		}
		return InnerType::template get<totalRank, offset + Index::rank>(body(subderef), deref);
	}

	template<int totalRank, int offset>
	static ScalarType const &get_const(BodyType const &body, intN<totalRank> const &deref) {
		intN<Index::rank> subderef;
		for (int i = 0; i < Index::rank; ++i) {
			subderef(i) = deref(i + offset);
		}
		return InnerType::template get_const<totalRank, offset + Index::rank>(body(subderef), deref);
	}

	//convert a write index (sizes by nesting depth, index values form 0 to memory sizes)
	// into read index (sized by rank, index values from 0 to index dimension)
	template<
		int totalRank,
		int currentRank,
		int numNestings,
		int currentNesting>
	static void getReadIndexForWriteIndex(
		intN<totalRank> &index,
		intN<numNestings> const &writeIndex)
	{
		intN<Index::rank> subIndex = BodyType::getReadIndexForWriteIndex(writeIndex(currentNesting));
		for (int i = 0; i < Index::rank; ++i) {
			index(i + currentRank) = subIndex(i);
		}
		InnerType::template getReadIndexForWriteIndex<
			totalRank,
			currentRank + Index::rank,
			numNestings,
			currentNesting + 1>
				(index, writeIndex);
	}
};

template<typename ScalarType_, typename Index_>
struct TensorStats<ScalarType_, Index_> {
	using ScalarType = ScalarType_;
	using Index = Index_;
	using InnerType = TensorStats<ScalarType>;
	static constexpr auto rank = Index::rank;
	
	using BodyType = typename Index::template Body<ScalarType, ScalarType>;

		//second-to-base (could be base) case
	
	//get an element in a tensor
	template<int totalRank, int offset>
	static ScalarType &get(BodyType &body, intN<totalRank> const &deref) {
		intN<Index::rank> subderef;
		for (int i = 0; i < Index::rank; ++i) {
			subderef(i) = deref(i + offset);
		}
		return body(subderef);
	}

	template<int totalRank, int offset>
	static ScalarType const &get_const(BodyType const &body, intN<totalRank> const &deref) {
		intN<Index::rank> subderef;
		for (int i = 0; i < Index::rank; ++i) {
			subderef(i) = deref(i + offset);
		}
		return body(subderef);
	}

	template<
		int totalRank,
		int currentRank,
		int numNestings,
		int currentNesting>
	static void getReadIndexForWriteIndex(
		intN<totalRank> &index,
		intN<numNestings> const &writeIndex)
	{
		intN<Index::rank> subIndex = BodyType::getReadIndexForWriteIndex(writeIndex(currentNesting));
		for (int i = 0; i < Index::rank; ++i) {
			index(i + currentRank) = subIndex(i);
		}
	}
};

//appease the vararg specialization recursive reference
//(it doesn't seem to recognize the single-entry base case)
template<typename ScalarType_>
struct TensorStats<ScalarType_> {
	using ScalarType = ScalarType_;
	static constexpr auto rank = 0;

	using BodyType = ScalarType;

	struct NullType {
		using InnerType = NullType;
		using BodyType = int;
		static constexpr auto rank = 0;
		using Index = NullType;
	};

	//because the fixed-number-of-ints dereferences are members of the tensor
	//we need safe ways for the compiler to reference those nested types
	//even when the tensor does not have a compatible rank
	//this seems like a bad idea
	using InnerType = NullType;
	using Index = NullType;

		//base case: do nothing
	
		//get an element in a tensor
	
	template<int totalRank, int offset>
	static ScalarType &get(BodyType &body, intN<totalRank> const &deref) {}
	
	template<int totalRank, int offset>
	static ScalarType const &get_const(BodyType const &body, intN<totalRank> const &deref) {}

	template<int totalRank, int currentRank, int numNestings, int currentNesting>
	static void getReadIndexForWriteIndex(intN<totalRank> &index, intN<numNestings> const &writeIndex) {}
};

/*
retrieves stats for a particular index of the tensor
currently stores dim

how it works:
if index <= TensorStats::Index::rank then
	use TensorStats::Index
otherwise
	perform the operation on
		index - rank, TensorStats::InnerType
*/
template<int index, typename TensorStats>
struct IndexStats {
	static constexpr auto dim = std::conditional<index < TensorStats::Index::rank,
		typename TensorStats::Index,
		IndexStats<index - TensorStats::Index::rank, typename TensorStats::InnerType>
	>::type::dim;
};

template<int writeIndex, typename TensorStats>
struct WriteIndexInfo {
	static constexpr auto size = std::conditional<writeIndex == 0,
		typename TensorStats::BodyType,
		WriteIndexInfo<writeIndex - 1, typename TensorStats::InnerType>
	>::type::size;
};

/*
Type		= tensor element type
TensorStats	= helper class for calculating some template values
BodyType 	= the BodyType (the generic_vector subclass) of the tensor
DerefType 	= the dereference type that would be needed to dereference this BodyType
				= int vector with dimension equal to rank

rank 		= total rank of the structure
examples:
	3-element vector:					tensor<double,upper<3>>
	4-element one-form:					tensor<double,lower<4>>
	matrix (contravariant matrix):		tensor<double,upper<3>,upper<3>>
	metric tensor (symmetric covariant matrix):	tensor<double, symmetric<lower<3>, lower<3>>>
*/
template<typename ScalarType_, typename... Args_>
struct Tensor {
	using Type = ScalarType_;
	using Args = std::tuple<Args_...>;

	//TensorStats metaprogram calculates rank
	//it pulls individual entries from the index args
	//I could have the index args themselves do the calculation
	// but that would mean making base-case specializations for each index class
	using TensorStats = ::Tensor::TensorStats<ScalarType_, Args_...>;
	using BodyType = typename TensorStats::BodyType;

	//used to get information per-index of the tensor
	// currently supports: dim
	// so Tensor<Real, Upper<2>, Symmetric<Upper<3>, Upper<3>>> can call ::IndexInfo<0>::dim to get 2,
	//  or call ::IndexInfo<1>::dim or ::IndexInfo<2>::dim to get 3
	template<int index>
	using IndexInfo = IndexStats<index, TensorStats>;

	static constexpr auto rank = TensorStats::rank;
	
	using DerefType = intN<rank>;

	//number of args deep
	static constexpr auto numNestings = std::tuple_size_v<Args>;

	//pulls a specific arg to get info from it
	// so Tensor<Real, Upper<2>, Symmetric<Upper<3>, Upper<3>>> can call ::WriteIndexInfo<0>::size to get 2,
	//  or call ::WriteIndexInfo<1>::size to get 6 = n*(n+1)/2 for n=3
	template<int writeIndex>
	using WriteIndexInfo = ::Tensor::WriteIndexInfo<writeIndex, TensorStats>;
		
	using WriteDerefType = intN<numNestings>;

	//here's a question
	// const_iterator cycles through all readable indexes
	//  and is const access so it can only be used for reading
	// Write::iterator cycles through all writeable indexes
	//  and is non-const access so it can be used for writing
	//   (actually it just returns indexes, so access doesn't matter)
	//  and it potentially skips elements so it is no good for reading
	// iterator cycles through all readable lndexes
	//   and is non-const access, so it can be used for writing
	//   but who would, when they would be writing over redundant indexes?
	//so should I replace iterator with Write::iterator?

	//read iterator
	
	//maybe I should put this in body
	//and then use a sort of nested iterator so it doesn't cover redundant elements in symmetric indexes
	struct iterator {
		Tensor *parent;
		DerefType index;
		
		iterator() : parent(NULL) {}
		iterator(Tensor *parent_) : parent(parent_) {}
		iterator(iterator const &iter) : parent(iter.parent), index(iter.index) {}
		
		bool operator==(iterator const &b) const { return index == b.index; }
		bool operator!=(iterator const &b) const { return index != b.index; }

		
		template<int i>
		struct Increment {
			static bool exec(iterator &iter) {
				++iter.index(i);
				if (iter.index(i) < IndexInfo<i>::dim) return true;
				if (i < rank-1) iter.index(i) = 0;
				return false;
			}
		};

		iterator &operator++() {
			Common::ForLoop<0,rank,Increment>::exec(*this);
			return *this;
		}

		Type &operator*() const { return (*parent)(index); }
	};

	iterator begin() {
		iterator i(this);
		return i;
	}
	iterator end() {
		iterator i(this);
		i.index(rank-1) = size()(rank-1);
		return i;
	}

	//maybe I should put this in body
	//and then use a sort of nested iterator so it doesn't cover redundant elements in symmetric indexes
	struct const_iterator {
		Tensor const *parent;
		DerefType index;
		
		const_iterator() : parent(NULL) {}
		const_iterator(Tensor const *parent_) : parent(parent_) {}
		const_iterator(const_iterator const &iter) : parent(iter.parent), index(iter.index) {}
		
		bool operator==(const_iterator const &b) const { return index == b.index; }
		bool operator!=(const_iterator const &b) const { return index != b.index; }
	
		template<int i>
		struct Increment {
			static bool exec(const_iterator &iter) {
				++iter.index(i);
				if (iter.index(i) < IndexInfo<i>::dim) return true;
				if (i < rank-1) iter.index(i) = 0;
				return false;
			}
		};

		const_iterator &operator++() {
			Common::ForLoop<0,rank,Increment>::exec(*this);
			return *this;
		}

		Type const &operator*() const { return (*parent)(index); }
	};

	const_iterator begin() const {
		const_iterator i(this);
		return i;
	}
	const_iterator end() const {
		const_iterator i(this);
		i.index(rank-1) = size()(rank-1);
		return i;
	}

	//iterating across data members to be written

	struct Write {
		
		struct iterator {
			//this is an array as deep as the number of indexes that the tensor was built with
			//each value of the array iterates from 0 to that nesting's type's ::size
			//The ::size returns the number of elements used to represent the structure,
			// so for a 3x3 symmetric matrix ::size would give 6, for n*(n+1)/2 elements used.
			WriteDerefType writeIndex;
			
			iterator() {}
			iterator(iterator const &iter) : writeIndex(iter.writeIndex) {}
			
			bool operator==(iterator const &b) const { return writeIndex == b.writeIndex; }
			bool operator!=(iterator const &b) const { return writeIndex != b.writeIndex; }

			//I could've for-loop'd this, but then I wouldn't have compile-time access to the write index size!
			template<int i>
			struct Increment {
				static bool exec(iterator& iter) {
					++iter.writeIndex(i);
					if (iter.writeIndex(i) < WriteIndexInfo<i>::size) return true;
					if (i < numNestings-1) iter.writeIndex(i) = 0;
					return false;
				}
			};

			iterator &operator++() {
				Common::ForLoop<0,numNestings,Increment>::exec(*this);
				return *this;
			}

			DerefType operator*() {
				return getReadIndexForWriteIndex(writeIndex);
			}

			DerefType getReadIndex() { return getReadIndexForWriteIndex(writeIndex); }

			static DerefType getReadIndexForWriteIndex(WriteDerefType const writeIndex) {
				DerefType index;
				TensorStats::template getReadIndexForWriteIndex<rank, 0, numNestings, 0>(index, writeIndex);
				return index;
			}
		};

		iterator begin() {
			return iterator();
		}

		iterator end() {
			iterator i;
			i.writeIndex(numNestings-1) = WriteIndexInfo<numNestings-1>::size;
			return i;
		}
	};

	//no longer needs any member data from the parent
	Write write() {
		return Write();
	}

	//constructors

	Tensor() {}
	Tensor(BodyType const & body_) : body(body_) {}
	Tensor(Tensor const & t) : body(t.body) {}

	//constructor from lambda

	Tensor(std::function<Type(DerefType)> f) {
		typename Write::iterator end = write().end();
		for (typename Write::iterator i = write().begin(); i != end; ++i) {
			DerefType index = i.getReadIndex();
			(*this)(index) = f(index);
		}
	}

	//constructor from members

	Tensor(
		typename TensorStats::InnerType::BodyType const & x0
	)
	requires (IndexInfo<0>::dim == 1)
	{
		body.v[0] = x0;
	}
	
	Tensor(
		typename TensorStats::InnerType::BodyType const & x0,
		typename TensorStats::InnerType::BodyType const & x1
	)
	requires (IndexInfo<0>::dim == 2)
	{
		body.v[0] = x0;
		body.v[1] = x1;
	}
	
	Tensor(
		typename TensorStats::InnerType::BodyType const & x0,
		typename TensorStats::InnerType::BodyType const & x1,
		typename TensorStats::InnerType::BodyType const & x2
	)
	requires (IndexInfo<0>::dim == 3)
	{
		body.v[0] = x0;
		body.v[1] = x1;
		body.v[2] = x2;
	}

	Tensor(
		typename TensorStats::InnerType::BodyType const & x0,
		typename TensorStats::InnerType::BodyType const & x1,
		typename TensorStats::InnerType::BodyType const & x2,
		typename TensorStats::InnerType::BodyType const & x3
	)
	requires (IndexInfo<0>::dim == 4)
	{
		body.v[0] = x0;
		body.v[1] = x1;
		body.v[2] = x2;
		body.v[3] = x3;
	}

	/*
	Tensor<real, Upper<3>> v;
	v.body(0) will return of type real

	Tensor<real, Upper<3>, upper<4>> v;
	v.body(0) will return of type upper<4>::body<real, real>
	*/
	BodyType body;

	// dereference by vararg ints

	template<int offset, typename... Rest>
	struct BuildDeref;

	template<int offset, typename... Rest>
	struct BuildDeref<offset, int, Rest...> {
		static DerefType exec(int next, Rest ... rest) {
			DerefType index = BuildDeref<offset+1, Rest...>::exec(rest...);
			index(offset) = next;
			return index;
		}
	};

	template<int offset>
	struct BuildDeref<offset, int> {
		static DerefType exec(int last) {
			static_assert(offset == rank-1, "didn't provide enough arguments for dereference");
			DerefType index;
			index(offset) = last;
			return index;
		}
	};

	template<typename... Rest>
	Type &operator()(int first, Rest... rest) {
		return (*this)(BuildDeref<0, int, Rest...>::exec(first, rest...));
	}

	template<typename... Rest>
	Type const &operator()(int first, Rest... rest) const {
		return (*this)(BuildDeref<0, int, Rest...>::exec(first, rest...));
	}

	//a single dereference using []
	// TODO this does NOT regard multi-rank storage like Symmetric
	// in order to do that, you need a(nother?) wrapper
	// though you could get by with this so long as the storage was rank-1 (i.e. Lower or Upper alone)

	BodyType::Type & operator[](int i) {
		return body[i];
	}

	BodyType::Type const & operator[](int i) const {
		return body[i];
	}

	//dereference by a vector of ints

	Type & operator()(DerefType const & deref) {
		return TensorStats::template get<rank,0>(body, deref);
	}
	
	Type const & operator()(DerefType const & deref) const {
		return TensorStats::template get_const<rank,0>(body, deref);
	}

	Tensor operator-() const { return Tensor(-body); }
	Tensor operator+(Tensor const &b) const { return Tensor(body + b.body); }
	Tensor operator-(Tensor const &b) const { return Tensor(body - b.body); }
	Tensor operator*(Type const &b) const { return Tensor(body * b); }
	Tensor operator/(Type const &b) const { return Tensor(body / b); }
	Tensor &operator+=(Tensor const &b) { body += b.body; return *this; }
	Tensor &operator-=(Tensor const &b) { body -= b.body; return *this; }
	Tensor &operator*=(Type const &b) { body *= b; return *this; }
	Tensor &operator/=(Type const &b) { body /= b; return *this; }

	template<int index>
	struct AssignSize {
		static bool exec(DerefType& input) {
			//now to make this nested type in Tensor ...
			input(index) = IndexInfo<index>::dim;
			return false;
		}
	};

	DerefType size() const {
		DerefType s;
		//metaprogram-driven
		Common::ForLoop<0, rank, AssignSize>::exec(s);
		return s;
	};

	//equality

	template<typename T>
	bool operator==(T const & t) const {
		if (rank != t.rank) return false;
		if (size() != t.size()) return false;
		
		//TODO write-iterator if they have matching symmetries
		for (const_iterator i = begin(); i != end(); ++i) {
			if (*i != t(i.index)) return false;
		}
		return true;
	}

	template<typename T>
	bool operator!=(T const & t) const {
		return !operator==(t);
	}

	//assignment

	template<typename T>
	Tensor& operator=(T const & t) {
		iterator i = begin();
		typename T::iterator j = t.read().begin();
		for (; i != end(); ++i, ++j) {
			if (j == t.end()) break;
			*i = *j;
		}
		return *this;
	}

	/*
	casting-to-body operations
	
	these are currently used when someone dereferences a portion of the tensor like so:
	Tensor<real, lower<dim>, lower<dim>> diff;
	diff(i) = (cell[index+dxi(i)].w(j) - cell[index-dxi(i)].w(j)) / (2 * dx(i))
	
	this is kind of abusive of the whole class.
	other options to retain this ability include wrapping returned tensor portions in tensor-like(-subclass?) accessors.
	
	the benefit of doing this is to allow assignments on tensors that have arbitrary rank.
	if I were to remove this then I would need some sort of alternative to do just that.
	this is where better iterators could come into play.
	*/
	operator BodyType&() { return body; }
	operator BodyType const&() const { return body; }

	//index assignment
	//t(i,j) = t(j,i) etc

	template<typename IndexType, typename... IndexTypes>
	IndexAccess<Tensor, std::tuple<IndexType, IndexTypes...>> operator()(IndexType, IndexTypes...) {
		return IndexAccess<Tensor, std::tuple<IndexType, IndexTypes...>>(this);
	}
};


template<typename Type, typename... Args>
std::ostream &operator<<(std::ostream &o, Tensor<Type, Args...> const &t) {
	using Tensor = Tensor<Type, Args...>;
	using const_iterator = typename Tensor::const_iterator;
	static constexpr auto rank = Tensor::rank;
	char const *empty = "";
	char const *sep = ", ";
	Vector<char const *,rank> seps(empty);
	for (const_iterator i = t.begin(); i != t.end(); ++i) {
		o << seps(0);
		for (int j = 0; j < rank; ++j) {
			bool matches = true;
			for (int k = 0; k < rank - j; ++k) {
				if (i.index(k) != 0) {
					matches = false;
					break;
				}
			}
			if (matches) o << "(";
		}
		o << *i;
		seps(0) = sep;
		for (int j = 0; j < rank; ++j) {
			bool matches = true;
			for (int k = 0; k < rank - j; ++k) {
				if (i.index(k) != t.size()(k)-1) {
					matches = false;
					break;
				}
			}
			if (matches) o << ")";
		}
	}
	return o;
}

}
