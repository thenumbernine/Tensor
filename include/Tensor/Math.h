#pragma once

// tensor math functions dependent on the tensors
// but also dependent on other things so I'm going to wedge in here some extra .h.h includes
// TODO call this "Tensor.h" and make use of Tensor.h?

#include "Tensor/Math.h.h"	// forward-declarations better match
#include "Tensor/Vector.h"	// class bodies must come first so I can use them
#include "Tensor/Range.h"	// class bodies must come first so I can use them

namespace Tensor {

//element-wise multiplication
// c_i1_i2_... := a_i1_i2_... * b_i1_i2_...
// Hadamard product / per-element multiplication
// TODO let 'a' and 'b' be dif types, so long as rank and dim match i.e. so long as the read-iterator domain matches.
//  pick the result to be the generalization of the two, so if one has sym<> indexes and the other doesn't use the not-sym in the result
template<typename T>
requires is_tensor_v<T>
T elemMul(T const & a, T const & b) {
	return T([&](typename T::intN i) -> typename T::Scalar {
		return a(i) * b(i);
	});
}

// GLSL naming compat
// TODO only for matrix? meh?
template<typename... T>
auto matrixCompMult(T&&... args) {
	return elemMul(std::forward<T>(args)...);
}

// more name compat
template<typename... T>
auto hadamard(T&&... args) {
	return elemMul(std::forward<T>(args)...);
}

// dot product.  To generalize this I'll consider it to be the Frobenius norm, since * will already be contraction
// TODO get rid of member functions  ...
// .... or have the member functions call into these.
// 	c := Σ_i1_i2_... a_i1_i2_... * b_i1_i2_...
template<typename A, typename B>
requires IsBinaryTensorOp<A,B>
typename A::Scalar inner(A const & a, B const & b) {
	auto i = a.begin();
	auto sum = a(i.index) * b(i.index);
	for (++i; i != a.end(); ++i) {
		sum += a(i.index) * b(i.index);
	}
	return sum;
}

// naming compat
template<typename... T>
auto dot(T&&... args) {
	return inner(std::forward<T>(args)...);
}

template<typename T>
requires is_tensor_v<T>
typename T::Scalar lenSq(T const & v) {
	return dot(v, v);
}

template<typename T>
requires (is_tensor_v<T>)
typename T::Scalar length(T const & v) {
	// TODO should I recast to Scalar, or just let it preserve double, or use sqrtf or what?
	return (typename T::Scalar)sqrt(lenSq(v));
}

template<typename A, typename B>
requires IsBinaryTensorOp<A,B>
typename A::Scalar distance(A const & a, B const & b) {
	return length(b - a);
}

template<typename T>
requires (is_tensor_v<T>)
T normalize(T const & v) {
	return v / length(v);
}

// c_i := ε_ijk * b_j * c_k
template<typename A, typename B>
requires IsBinaryTensorR3xR3Op<A,B>
auto cross(A const & a, B const & b) {
	return A(
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]);
}

// outer product of tensors c_i1_..ip_j1_..jq = a_i1..ip * b_j1..jq
// for vectors: c_ij := a_i * b_j
template<typename A, typename B>
requires IsBinaryTensorOp<A,B>
auto outer(A const & a, B const & b) {
	using AB = typename A::template ReplaceScalar<B>;
	//another way to implement would be a per-elem .map(), and just return the new elems as a(i) * b
	return AB([&](typename AB::intN i) -> typename A::Scalar {
		static_assert(decltype(i)::template dim<0> == A::rank + B::rank);
		return a(i.template subset<A::rank, 0>()) * b(i.template subset<B::rank, A::rank>());
	});
}

// GLSL naming compat
template<typename... T>
auto outerProduct(T&&... args) {
	return outer(std::forward<T>(args)...);
}

// matrix functions

#if 0
//https://stackoverflow.com/a/50471331
template<typename T, std::size_t N, typename... Ts>
constexpr std::array<T, N> permuteArray(
	std::array<T, N> const & arr,
	std::array<int, N> const & permutation,
	Ts&&... processed
) {
	if constexpr (sizeof...(Ts) == N) {
		return std::array<T, N>{ std::forward<Ts>(processed)... };
	} else {
		return permute(
			arr,
			permutation,
			std::forward<Ts>(processed)...,
			arr[permutation[sizeof...(Ts)]]
		);
	}
}

//aka 'reshape' ?
// nah reshape in matlab is for resizing dimensions and maintaining storage.
// this is more of a swizzle (dimension-swap) but for ranks ... rank-swap ... index-swap ...
template<typename T, typename I, I... Is>
require (is_tensor_v<T>)
auto permuteIndexes(T const & t) {
	using R =
		T
			... for each i ...
			::template ReplaceDim<i, T::template dim<Is[i]>>;
	return R([](typename R::intN i) -> typename R::Scalar {
		return t(permuteArray(i));
	});
}
#endif

/*
Transpose, i.e. exchange two indexes

If the two indexes are sequential and _sym symmetric storage is used then nothing is changed.
TODO if the two indexes are sequential and _asym storage is used then just negative the values.
Otherwise expand the internal storage at indexes m and n (i.e. convert it from _sym or _asym into _vec),
then exchange the dimensions.
*/
template<int m/*=0*/, int n/*=1*/, typename T>
requires (
	is_tensor_v<T>
	&& T::rank >= 2
)
auto transpose(T const & t) {
	if constexpr (m == n) {
		//don't reshape if we're flipping the same index with itself
		return t;
	} else if constexpr (
		// don't reshape symmetric -- their transpose is identity
		T::template numNestingsToIndex<m> == T::template numNestingsToIndex<n>
		&& is_sym_v<T>
	) {
		return t;
	} else if constexpr (
		// don't reshape antisymmetric -- their transpose is negative
		T::template numNestingsToIndex<m> == T::template numNestingsToIndex<n>
		&& is_asym_v<T>
	) {
		return -t;
	} else {	// m < n and they are different storage nestings
		constexpr int mdim = T::template dim<m>;
		constexpr int ndim = T::template dim<n>;
		// now re-index E to exchange dimensions
		// replace the storage at index m with an equivalent one but of dimension of n's index
		using Tnm = typename T
			::template ExpandIndex<m, n> //ReplaceDim doesn't guarantee to expand if the dims match
			::template ReplaceDim<m, ndim>
			::template ReplaceDim<n, mdim>;
		return Tnm([&](typename Tnm::intN i) {
			std::swap(i(m), i(n));
			return t(i);
		});
	}
}

// contraction of two indexes of a tensor
template<int m/*=0*/, int n/*=1*/, typename T>
requires (is_tensor_v<T>
	&& m < T::rank
	&& n < T::rank
	&& T::template dim<m> == T::template dim<n>
)
auto contract(T const & t) {
	using S = typename T::Scalar;
	if constexpr (m > n) {
		//ensure m < n esp for swizzling index's sake
		return contract<n,m,T>(t);
	} else if constexpr (m == n) {
		// hmmmm Σ_i g^ii a_i is a technically-incorrect tensor operation
		if constexpr (T::rank == 1) {
			S sum = t(0);
			for (int k = 1; k < T::template dim<0>; ++k) {
				sum += t(k);
			}
			return sum;
		} else {
			using R = typename T::template RemoveIndex<m>;
			// TODO a macro to remove the m'th element from 'int... i'
			//return R([](auto... is) -> S {
			// or TODO implement intN access to asym (and fully sym)
			return R([&](typename R::intN i) -> S {
				// static_assert R::intN::dims == T::intN::dims-1
				auto j = typename T::intN([&](int jk) -> int {
					if (jk == m) return 0; // set j[m] = 0
					if (jk > m) --jk;
					return i[jk];
				});
				//j[m] == 0 ...
				S sum = t(j);
				for (int k = 1; k < T::template dim<m>; ++k) {
					j[m] = k;
					sum += t(j);
				}
				return sum;
			});
		}
	} else { // m < n
		if constexpr (T::rank == 2) {
			S sum = t(0,0);
			for (int k = 1; k < T::template dim<m>; ++k) {
				sum += t(k,k);
			}
			return sum;
		} else {
			using R = typename T::template RemoveIndex<m,n>;
			return R([&](typename R::intN i) -> S {
				// static_assert R::intN::dims == T::intN::dims-2
				auto j = typename T::intN([&](int jk) -> int {
					if (jk == m) return 0; // j[m] = 0
					if (jk == n) return 0; // j[n] = 0
					// make sure you compare in decreasing order or else a decrement can destroy iteration and lead to oob lookups
					if (jk > n) --jk; // if m != n then do this twice
					if (jk > m) --jk; //  but (case above) if m == n do this once
					return i[jk];
				});
				// j[m] = j[n] = 0 ...
				S sum = t(j);
				for (int k = 1; k < T::template dim<m>; ++k) {
					j[m] = j[n] = k;
					sum += t(j);
				}
				return sum;
			});
		}
	}
}

// naming compat
template<int m, int n, typename T>
auto trace(T const & t) {
	return contract<m,n,T>(t);
}

//contracts the first index with the next count index and repeat count times
template<int index, int count, typename A>
requires (is_tensor_v<A>)
auto contractN(A const & a) {
	static_assert(index >= 0 && index < A::rank);
	static_assert(index + count >= 0 && index + count < A::rank);
	if constexpr (count == 0) {
		return a;
	} else {
		auto ac = contract<index,index+count>(a);
		if constexpr (count == 1) {
			return ac;
		} else {
			return contractN<index,count-1>(ac);
		}
	}
}

// this isn't really interior, but more of a mix of interior + outer + contract
// it is an interior product provided the num. of indexes == A::rank
// it is matrix-mul if num == 1
// TODO this could stand to be optimized
//OwnerRef is not really needed ... how about I just pass lambdas?
// but will lambdas constexpr?
template<int num, typename A, typename B>
requires IsInteriorOp<num, A, B>
auto interior(A const & a, B const & b) {
#if 0
	return contractN<A::rank-num,num>(outer(a,b));
#else
	using S = typename A::Scalar;
	if constexpr (A::rank == num && B::rank == num) {
		// rank-0 i.e. scalar result case
		static_assert(A::dims() == B::dims());	//thanks to the 3rd requires condition
		return dot(a,b);
	} else {
		using R = typename A
			::template ReplaceScalar<B>
			::template RemoveIndexSeq<Common::make_integer_range<int, A::rank-num, A::rank+num>>;
		static_assert(num != 1 || std::is_same_v<R, decltype(contract<A::rank-1,A::rank>(outer(a,b)))>);
		static_assert(std::is_same_v<R, decltype(contractN<A::rank-num,num>(outer(a,b)))>);
		return R([&](typename R::intN i) -> S {
			auto ai = typename A::intN([&](int j) -> int {
				if (j >= A::rank-num) return 0;
				return i[j];
			});
			auto bi = typename B::intN([&](int j) -> int {
				if (j < num) return 0;
				j += A::rank-2*num;
				return i[j];
			});
			
			//TODO instead use A::dim<A::rank-num..A::rank>
			S sum = {};
#if 0
			template<typename B>
			struct InteriorRangeIter {
				template<int i> constexpr int getRangeMin() const { return 0; }
				template<int i> constexpr int getRangeMax() const { return B::dims().template dim<i>; }
			};
			for (auto k : RangeIteratorInner<num, InteriorRangeIter<B>>(InteriorRangeIter<B>())) {
#else
			for (auto k : RangeObj<num, false>(_vec<int, num>(), B::dims().template subset<num, 0>())) {
#endif
				std::copy(k.s.begin(), k.s.end(), ai.s.begin() + (A::rank - num));
				std::copy(k.s.begin(), k.s.end(), bi.s.begin());
				sum += a(ai) * b(bi);
			}
			return sum;
		});
	}
#endif
}

// symmetrize or antisymmetrize a tensor
//  I am not convinced this should be the default casting operation from non-(a)sym to (a)sym since it incurs a few more operations
// but it should def be made available

template<typename T>
requires IsSquareTensor<T>
auto makeSym(T const & t) {
	using S = typename T::Scalar;
	using intN = typename T::intN;
	using R = std::conditional_t<T::rank == 2,
		_sym<S, T::template dim<0>>,
		_symR<S, T::template dim<0>, T::rank>>;
	// iterate over write index, then iterate over all permutations of the read index and sum
	return R([&](intN i) -> S {
		S result = {};
		//'j' is our permutation
		intN j = initIntVecWithSequence<std::make_integer_sequence<int, T::rank>>::value();
		do {
			// 'ij' is 'i' permuted by 'j'
			intN ij = [&](int k) -> int { return i[j[k]]; };
			result += t(ij);
		} while (std::next_permutation(j.s.begin(), j.s.end()));
		return result / (S)factorial(T::rank);
	});
}

//that's right, same function, just different return type
template<typename T>
requires IsSquareTensor<T>
auto makeAsym(T const & t) {
	using S = typename T::Scalar;
	using intN = typename T::intN;
	using R = std::conditional_t<T::rank == 2,
		_asym<S, T::template dim<0>>,
		_asymR<S, T::template dim<0>, T::rank>>;
	// iterate over write index, then iterate over all permutations of the read index and sum
	return R([&](intN i) -> S {
		S result = {};
		//'j' is our permutation
		intN j = initIntVecWithSequence<std::make_integer_sequence<int, T::rank>>::value();
		do {
			//count # of flips
			// TODO combine this with next_permutation so you don't have to keep recounting
			intN sortedj = j;
			auto sign = antisymSortAndCountFlips(sortedj);
			if (sign == Sign::ZERO) {
				throw Common::Exception() << "shouldn't get here";
			}
			// 'ij' is 'i' permuted by 'j'
			intN ij = [&](int k) -> int { return i[j[k]]; };
			if (sign == Sign::NEGATIVE) {
				result -= t(ij);
			} else {
				result += t(ij);
			}
		} while (std::next_permutation(j.s.begin(), j.s.end()));
		return result / (S)factorial(T::rank);
	});
}

// wedge product

template<typename A, typename B>
requires IsBinaryTensorOp<A,B>
auto wedge(A const & a, B const & b) {
	return makeAsym(outer(a,b));
}

// Hodge dual

template<typename A>
requires IsSquareTensor<A>
auto hodgeDual(A const & a) {
	static constexpr int rank = A::rank;
	static constexpr int dim = A::template dim<0>;
	static_assert(rank <= dim);
	using S = typename A::Scalar;
	return interior<rank>(a, _asymR<S, dim, dim>(1)) / (S)factorial(rank);
}

// operator* is contract(outer(a,b)) ... aka interior1(a,b)
// TODO maybe generalize further with the # of indexes to contract:
// c_i1...i{p}_j1_..._j{q} = Σ_k1...k{r} a_i1_..._i{p}_k1_...k{r} * b_k1_..._k{r}_j1_..._j{q}

template<typename A, typename B>
requires IsBinaryTensorOpWithMatchingNeighborDims<A,B>
auto operator*(A const & a, B const & b) {
#if 1	// lazy way.  inline the lambdas and don't waste the outer()'s operation expenses
	//return contract<A::rank-1, A::rank>(outer(a,b));
	return interior<1>(a,b);
#else	// some optimizations, no wasted storage
	using S = typename A::Scalar;
	if constexpr (A::rank == 1 && B::rank == 1) {
		// rank-0 result case
		static_assert(A::dims() == B::dims());	//thanks to the 3rd requires condition
		//scalar return case
		S sum = a(0) * b(0);
		for (int k = 1; k < A::template dim<0>; ++k) {
			sum += a(k) * b(k);
		}
		return sum;
	} else {
		using R = typename A
			::template ReplaceScalar<B>
			::template RemoveIndex<A::rank-1, A::rank>;
		//static_assert(std::is_same_v<typename A::template ReplaceScalar<B>, decltype(outer(a,b))>);
		//static_assert(std::is_same_v<R, decltype(interior<1>(a,b))>);
		return R([&](typename R::intN i) -> S {
			auto ai = typename A::intN([&](int j) -> int {
				if (j == A::rank-1) return 0;
				return i[j];
			});
			auto bi = typename B::intN([&](int j) -> int {
				if (j == 0) return 0;
				j += A::rank-2;
				return i[j];
			});
			S sum = a(ai) * b(bi);
			for (int k = 1; k < A::template dim<A::rank-1>; ++k) {
				ai(A::rank-1) = k;
				bi(0) = k;
				sum += a(ai) * b(bi);
			}
			return sum;
		});
	}
#endif
}

// diagonalize an index
// well if it's diagonal, might as well use _sym
template<int m/*=0*/, typename T>
requires (is_tensor_v<T>)
auto diagonal(T const & t) {
	static_assert(m >= 0 && m < T::rank);
	/* TODO ::InsertIndex that uses _tensor index indicators for what kind of storage to insert
	using R = T
		::ExpandIndex<m>
		::InsertIndex<m, index_vec<T::dim<m>>>;
	... but can't use InsertIndex to insert a _sym with the expanded index
	... maybe instead
	using R = T
		::RemoveIndex<m>
		::InsertIndex<m, index_sym<T::dim<m>>>
	*/
	using E = typename T::template ExpandIndex<m>;
	constexpr int nest = E::template numNestingsToIndex<m>;
	using R = typename E
		::template ReplaceNested<
			nest,
			_sym<
				typename E::template Nested<nest+1>,
				E::template dim<m>
			>
		>;
	return R([&](typename R::intN i) {
		if (i[m] != i[m+1]) return typename T::Scalar();
		auto isrc = typename T::intN([&](int j) {
			if (j >= m) return i[j+1];
			return i[j];
		});
		return t(isrc);
	});
}

}
