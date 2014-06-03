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
	typedef ScalarType_ ScalarType;
	typedef Index_ Index;
	typedef TensorStats<ScalarType, Args...> InnerType;
	
	//rank of the tensor.
	//vector indexes contribute 1's, matrix (sym & antisym) contribute 2, etc
	enum { rank = Index::rank + InnerType::rank };		
	
	//nestings of the tensor.
	//number of InnerTypes deep we can go
	//if it has two vector indexes then this gets 2.  if it has one sym mat then its 1, etc.
	enum { numNestings = 1 + InnerType::numNestings };	

	typedef typename Index::template Body<typename InnerType::BodyType, ScalarType> BodyType;
	
		//inductive cases
	
	//get an element in a tensor
	template<int totalRank, int offset>
	static ScalarType &get(BodyType &body, const Vector<int,totalRank> &deref) {
		Vector<int,Index::rank> subderef;
		for (int i = 0; i < Index::rank; ++i) {
			subderef(i) = deref(i + offset);
		}
		return InnerType::template get<totalRank, offset + Index::rank>(body(subderef), deref);
	}

	template<int totalRank, int offset>
	static const ScalarType &get_const(const BodyType &body, const Vector<int,totalRank> &deref) {
		Vector<int,Index::rank> subderef;
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
		Vector<int, totalRank> &index,
		const Vector<int, numNestings> &writeIndex)
	{
		Vector<int,Index::rank> subIndex = BodyType::getReadIndexForWriteIndex(writeIndex(currentNesting));
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
	typedef ScalarType_ ScalarType;
	typedef Index_ Index;
	typedef TensorStats<ScalarType> InnerType;
	enum { rank = Index::rank };
	enum { numNestings = 1 };
	
	typedef typename Index::template Body<ScalarType, ScalarType> BodyType;

		//second-to-base (could be base) case
	
	//get an element in a tensor
	template<int totalRank, int offset>
	static ScalarType &get(BodyType &body, const Vector<int,totalRank> &deref) {
		Vector<int,Index::rank> subderef;
		for (int i = 0; i < Index::rank; ++i) {
			subderef(i) = deref(i + offset);
		}
		return body(subderef);
	}

	template<int totalRank, int offset>
	static const ScalarType &get_const(const BodyType &body, const Vector<int,totalRank> &deref) {
		Vector<int,Index::rank> subderef;
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
		Vector<int, totalRank> &index,
		const Vector<int, numNestings> &writeIndex)
	{
		Vector<int,Index::rank> subIndex = BodyType::getReadIndexForWriteIndex(writeIndex(currentNesting));
		for (int i = 0; i < Index::rank; ++i) {
			index(i + currentRank) = subIndex(i);
		}	
	}
};

//appease the vararg specialization recursive reference
//(it doesn't seem to recognize the single-entry base case)
template<typename ScalarType_>
struct TensorStats<ScalarType_> {
	typedef ScalarType_ ScalarType;
	enum { rank = 0 };
	enum { numNestings = 0 };

	typedef ScalarType BodyType;

	struct NullType {
		typedef NullType InnerType;
		typedef int BodyType;
		enum { rank = 0 };
		typedef NullType Index;
	};

	//because the fixed-number-of-ints dereferences are members of the tensor
	//we need safe ways for the compiler to reference those nested types
	//even when the tensor does not have a compatible rank
	//this seems like a bad idea
	typedef NullType InnerType;
	typedef NullType Index;

		//base case: do nothing
	
		//get an element in a tensor
	
	template<int totalRank, int offset>
	static ScalarType &get(BodyType &body, const Vector<int,totalRank> &deref) {}
	
	template<int totalRank, int offset>
	static const ScalarType &get_const(const BodyType &body, const Vector<int,totalRank> &deref) {}

	template<int totalRank, int currentRank, int numNestings, int currentNesting>
	static void getReadIndexForWriteIndex(Vector<int, totalRank> &index, const Vector<int, numNestings> &writeIndex) {}
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
	enum { 
		dim = If<index < TensorStats::Index::rank,
			typename TensorStats::Index,
			IndexStats<index - TensorStats::Index::rank, typename TensorStats::InnerType>
		>::Type::dim
	};
};

template<int writeIndex, typename TensorStats>
struct WriteIndexStats {
	enum {
		size = If<writeIndex == 0,
			typename TensorStats::BodyType,
			WriteIndexStats<writeIndex - 1, typename TensorStats::InnerType>
		>::Type::size
	};
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
template<typename ScalarType_, typename... Args>
struct Tensor {
	typedef ScalarType_ Type;

	//TensorStats metaprogram calculates rank
	//it pulls individual entries from the index args
	//I could have the index args themselves do the calculation
	// but that would mean making base-case specializations for each index class 
	typedef TensorStats<ScalarType_, Args...> TensorStats;
	typedef typename TensorStats::BodyType BodyType;

	//used to get information per-index of the tensor
	// currently supports: dim
	// so Tensor<Real, Upper<2>, Symmetric<Upper<3>, Upper<3>>> can call ::IndexInfo<0>::dim to get 2, 
	//  or call ::IndexInfo<1>::dim or ::IndexInfo<2>::dim to get 3
	template<int index>
	using IndexInfo = IndexStats<index, TensorStats>;

	enum { rank = TensorStats::rank };
	
	typedef Vector<int,rank> DerefType;

	//number of args deep
	enum { numNestings = TensorStats::numNestings };

	//pulls a specific arg to get info from it
	// so Tensor<Real, Upper<2>, Symmetric<Upper<3>, Upper<3>>> can call ::WriteIndexInfo<0>::size to get 2,
	//  or call ::WriteIndexInfo<1>::size to get 6 = n*(n+1)/2 for n=3
	template<int writeIndex>
	using WriteIndexInfo = WriteIndexStats<writeIndex, TensorStats>;
		
	typedef Vector<int,numNestings> WriteDerefType;

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
		iterator(const iterator &iter) : parent(iter.parent), index(iter.index) {}
		
		bool operator==(const iterator &b) const { return index == b.index; }
		bool operator!=(const iterator &b) const { return index != b.index; }
	
		struct Increment {
			typedef iterator& Input;
			template<int i>
			struct Exec {
				static bool exec(Input iter) {
					++iter.index(i);
					if (iter.index(i) < IndexInfo<i>::dim) return true;
					if (i < rank-1) iter.index(i) = 0;
					return false;
				}
			};
		};

		iterator &operator++() {
			ForLoop<0,rank,Increment>::exec(*this);
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
		const Tensor *parent;
		DerefType index;
		
		const_iterator() : parent(NULL) {}
		const_iterator(const Tensor *parent_) : parent(parent_) {}
		const_iterator(const const_iterator &iter) : parent(iter.parent), index(iter.index) {}
		
		bool operator==(const const_iterator &b) const { return index == b.index; }
		bool operator!=(const const_iterator &b) const { return index != b.index; }
	
		struct Increment {
			typedef const_iterator& Input;
			template<int i>
			struct Exec {
				static bool exec(Input iter) {
					++iter.index(i);
					if (iter.index(i) < IndexInfo<i>::dim) return true;
					if (i < rank-1) iter.index(i) = 0;
					return false;
				}
			};
		};

		const_iterator &operator++() {
			ForLoop<0,rank,Increment>::exec(*this);
			return *this;
		}

		const Type &operator*() const { return (*parent)(index); }
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
			iterator(const iterator &iter) : writeIndex(iter.writeIndex) {}
			
			bool operator==(const iterator &b) const { return writeIndex == b.writeIndex; }
			bool operator!=(const iterator &b) const { return writeIndex != b.writeIndex; }

			//I could've for-loop'd this, but then I wouldn't have compile-time access to the write index size!
			struct Increment {
				typedef iterator& Input;
				template<int i>
				struct Exec {
					static bool exec(Input iter) {
						++iter.writeIndex(i);
						if (iter.writeIndex(i) < WriteIndexInfo<i>::size) return true;
						if (i < numNestings-1) iter.writeIndex(i) = 0;
						return false;
					}
				};
			};

			iterator &operator++() {
				ForLoop<0,numNestings,Increment>::exec(*this);
				return *this;
			}

			DerefType operator*() {
				return getReadIndexForWriteIndex(writeIndex);
			}

			DerefType getReadIndex() { return getReadIndexForWriteIndex(writeIndex); }

			static DerefType getReadIndexForWriteIndex(const WriteDerefType writeIndex) {
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
	Tensor(const BodyType &body_) : body(body_) {}
	Tensor(const Tensor &t) : body(t.body) {}

	Tensor(std::function<Type(DerefType)> f) {
		typename Write::iterator end = write().end();
		for (typename Write::iterator i = write().begin(); i != end; ++i) {
			DerefType index = i.getReadIndex();
			(*this)(index) = f(index);
		}
	}
	
	/*
	Tensor<real, upper<3>> v;
	v.body(0) will return of type real

	Tensor<real, upper<3>, upper<4>> v;
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
	const Type &operator()(int first, Rest... rest) const {
		return (*this)(BuildDeref<0, int, Rest...>::exec(first, rest...));
	}

	//dereference by a vector of ints

	Type &operator()(const DerefType &deref) { 
		return TensorStats::template get<rank,0>(body, deref);
	}
	
	const Type &operator()(const DerefType &deref) const { 
		return TensorStats::template get_const<rank,0>(body, deref);
	}

	Tensor operator-() const { return Tensor(-body); }
	Tensor operator+(const Tensor &b) const { return Tensor(body + b.body); }
	Tensor operator-(const Tensor &b) const { return Tensor(body - b.body); }
	Tensor operator*(const Type &b) const { return Tensor(body * b); }
	Tensor operator/(const Type &b) const { return Tensor(body / b); }
	Tensor &operator+=(const Tensor &b) { body += b.body; return *this; }
	Tensor &operator-=(const Tensor &b) { body -= b.body; return *this; }
	Tensor &operator*=(const Type &b) { body *= b; return *this; }
	Tensor &operator/=(const Type &b) { body /= b; return *this; }

	struct AssignSize {
		typedef DerefType& Input;

		template<int index>
		struct Exec {
			static bool exec(Input input) {
				//now to make this nested type in Tensor ...
				input(index) = IndexInfo<index>::dim;
				return false;
			}
		};
	};

	DerefType size() const {
		DerefType s;
		//metaprogram-driven
		ForLoop<0, rank, AssignSize>::exec(s);
		return s;
	};

	//equality

	template<typename T>
	bool operator==(const T &t) const {
		if (rank != t.rank) return false;
		if (size() != t.size()) return false;
		for (const_iterator i = begin(); i != end(); ++i) {
			if (*i != t(i.index)) return false;
		}
		return true;
	}

	template<typename T>
	bool operator!=(const T &t) const {
		return !operator==(t);
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
	operator const BodyType&() const { return body; }

	//index assignment
	//t(i,j) = t(j,i) etc

	template<int offset, typename... Rest>
	struct BuildIndex;
	
	template<int offset, typename... Rest>
	struct BuildIndex<offset, Index, Rest...> {
		static Vector<Index*, rank> exec(Index &next, Rest & ... rest) {
			Vector<Index*, rank> indexes = BuildIndex<offset+1, Rest...>::exec(rest...);
			indexes(offset) = &next;
			return indexes;
		}
	};

	template<int offset>
	struct BuildIndex<offset, Index> {
		static Vector<Index*, rank> exec(Index &last) {
			static_assert(offset == rank-1, "didn't provide enough arguments for index operation");
			Vector<Index*, rank> indexes;
			indexes(offset) = &last;
			return indexes;
		}
	};

	template<typename... Rest>
	IndexAccess<Tensor> operator()(Index &first, Rest & ... rest) {
		return IndexAccess<Tensor>(this, BuildIndex<0, Index, Rest...>::exec(first, rest...));
	}
};


template<typename Type, typename... Args>
std::ostream &operator<<(std::ostream &o, const Tensor<Type, Args...> &t) {
	
	typedef Tensor<Type, Args...> Tensor;
	typedef typename Tensor::const_iterator const_iterator;
	enum { rank = Tensor::rank };
	typedef typename Tensor::DerefType DerefType;
	const char *empty = "";
	const char *sep = ", ";
	Vector<const char *,rank> seps(empty);
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

};

