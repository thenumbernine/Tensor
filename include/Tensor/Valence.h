#pragma once

#include <utility>	//integer_sequence
#include "Common/Sequence.h"

/*
Valence is a thin wrapper around tensor<>
except that when indexing it, it produces a valenceIndex thin wrapper around Index
*/

namespace Tensor {

#define TENSOR_VALENCE_ADD_VECTOR_OP_EQ(op)\
	constexpr This & operator op(This const & b) {\
		t op b.t;\
		return *this;\
	}

#define TENSOR_VALENCE_ADD_SCALAR_OP_EQ(op)\
	constexpr This & operator op(Scalar const & b) {\
		t op b;\
		return *this;\
	}

#define TENSOR_VALENCE_ADD_UNARY(op)\
	constexpr This operator op() const {\
		return This(op t);\
	}

template<typename T>
concept is_valence_v = T::isValenceFlag;

template<typename Tensor_, char ... Vs>
requires is_tensor_v<Tensor_>
struct valence {
	using This = valence;
	using Tensor = Tensor_;
	static constexpr bool isValenceFlag = true;
	using Scalar = typename Tensor::Scalar;
	static constexpr int rank = Tensor::rank;
	
	template<typename NewTensor>
	using ReplaceTensor = valence<NewTensor, Vs...>;

	Tensor t;
	valence() {}
	valence(Tensor const & t_) : t(t_) {}
	valence(Tensor && t_) : t(t_) {}

	using valseq = std::integer_sequence<char, Vs...>;
	static_assert(sizeof...(Vs) == Tensor::rank);

	template<int index>
	static constexpr char val = Common::seq_get_v<index, valseq>;

	decltype(auto) operator*() { return t; }
	decltype(auto) operator*() const { return t; }
	decltype(auto) operator->() { return t; }
	decltype(auto) operator->() const { return t; }

	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(+=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(-=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(*=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(/=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(<<=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(>>=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(&=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(|=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(^=)
	TENSOR_VALENCE_ADD_VECTOR_OP_EQ(%=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(+=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(-=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(*=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(/=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(<<=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(>>=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(&=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(|=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(^=)
	TENSOR_VALENCE_ADD_SCALAR_OP_EQ(%=)
	TENSOR_VALENCE_ADD_UNARY(-)
	TENSOR_VALENCE_ADD_UNARY(~)

	//TENSOR_ADD_VECTOR_CALL_INDEX_PRIMARY()
	template<typename Int>
	requires std::is_integral_v<Int>
	constexpr decltype(auto) operator()(Int const i) {
		return t(i);
	}
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int const i) const {
		return t(i);
	}

	//operator[](Int), operator()(Int...) fwd to operator()(Int)(...)
	TENSOR_ADD_RANK1_CALL_INDEX_AUX()
	//operator()(vec<Int,N>) fwd to operator()(Int...)
	TENSOR_ADD_INT_VEC_CALL_INDEX()

	// TODO Index based operator() and a whole set of wrappers of Index expression operators...

	// TODO member method forwarding ...?
	// or just rely on * -> operators?
	// nahhh, because those won't check and apply valence wrappers.
};

template<typename Tensor, typename Seq>
struct valenceSeqImpl;
template<typename Tensor, typename I, I i1, I... is>
struct valenceSeqImpl<Tensor, std::integer_sequence<I, i1, is...>> {
	using type = valence<Tensor, i1, is...>;
};
template<typename Tensor, typename Seq>
using valenceSeq = typename valenceSeqImpl<Tensor, Seq>::type;

// so you don't have to explicitly write the first template arg ...
template<char ... vs>
auto make_valence(auto const & v) {
	return valence<std::decay_t<decltype(v)>, vs...>(v);
}
template<char ... vs>
auto make_valence(auto && v) {
	return valence<std::decay_t<decltype(v)>, vs...>(v);
}

template<typename Seq>
auto makeValenceSeq(auto const & v) {
	return valenceSeq<std::decay_t<decltype(v)>, Seq>(v);
}
template<typename Seq>
auto makeValenceSeq(auto && v) {
	return valenceSeq<std::decay_t<decltype(v)>, Seq>(v);
}

template<typename A, typename B>
requires (
	is_valence_v<A> &&
	is_valence_v<B> &&
	std::is_same_v<typename A::valseq, typename B::valseq>	// compile-time check that the valence matches?
)
bool operator==(A const & a, B const & b) {
	return a.t == b.t;
}

template<typename A, typename B>
requires (
	is_valence_v<A> &&
	is_valence_v<B> &&
	std::is_same_v<typename A::valseq, typename B::valseq>	// compile-time check that the valence matches?
)
bool operator!=(A const & a, B const & b) {
	return !operator==(a,b);
}

#define TENSOR_VALENCE_SCALAR_OP(op)\
template<typename A, typename B>\
requires (is_valence_v<A> && !is_valence_v<B> && !is_tensor_v<B>)\
decltype(auto) operator op(A const & a, B const & b) {\
	return makeValenceSeq<typename A::valseq>(a.t op b);\
}\
template<typename A, typename B>\
requires (!is_valence_v<A> && !is_tensor_v<A> && is_valence_v<B>)\
decltype(auto) operator op(A const & a, B const & b) {\
	return makeValenceSeq<typename B::valseq>(a op b.t);\
}

TENSOR_VALENCE_SCALAR_OP(+)
TENSOR_VALENCE_SCALAR_OP(-)
TENSOR_VALENCE_SCALAR_OP(*)
TENSOR_VALENCE_SCALAR_OP(/)

// this is distinct because it needs the require ! ostream
#define TENSOR_VALENCE_SCALAR_SHIFT_OP(op)\
template<typename A, typename B>\
requires (\
	is_valence_v<A> &&\
	!is_valence_v<B> && !is_tensor_v<B> &&\
	!std::is_base_of_v<std::ios_base, std::decay_t<B>>\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	return makeValenceSeq<typename A::valseq>(a.t op b);\
}\
\
template<typename A, typename B>\
requires (\
	!is_valence_v<A> && !is_tensor_v<A> &&\
	!std::is_base_of_v<std::ios_base, std::decay_t<A>> &&\
	is_valence_v<B>\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	return makeValenceSeq<typename B::valseq>(a op b.t);\
}

#define TENSOR_VALENCE_VALENCE_OP(op)\
template<typename A, typename B>\
requires (\
	is_valence_v<A> && is_valence_v<B> &&\
	std::is_same_v<typename A::valseq, typename B::valseq>\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	return makeValenceSeq<typename A::valseq>(a.t op b.t);\
}

TENSOR_VALENCE_VALENCE_OP(+)
TENSOR_VALENCE_VALENCE_OP(-)
TENSOR_VALENCE_VALENCE_OP(/)

TENSOR_VALENCE_VALENCE_OP(<<)
TENSOR_VALENCE_VALENCE_OP(>>)
TENSOR_VALENCE_VALENCE_OP(&)
TENSOR_VALENCE_VALENCE_OP(|)
TENSOR_VALENCE_VALENCE_OP(^)
TENSOR_VALENCE_VALENCE_OP(%)

TENSOR_VALENCE_SCALAR_SHIFT_OP(<<)
TENSOR_VALENCE_SCALAR_SHIFT_OP(>>)
TENSOR_VALENCE_SCALAR_OP(&)
TENSOR_VALENCE_SCALAR_OP(|)
TENSOR_VALENCE_SCALAR_OP(^)
TENSOR_VALENCE_SCALAR_OP(%)

//not yet in tensor:
//TENSOR_UNARY_OP(~)
//TENSOR_UNARY_OP(!)
//TENSOR_TENSOR_OP(&&)
//TENSOR_TENSOR_OP(||)
//TENSOR_TERNARY_OP(?:)

template<typename A, typename B>
requires is_valence_v<A> && is_valence_v<B>
auto operator*(A const & a, B const & b) {
	static_assert(A::template val<A::rank-1> != B::template val<0>);
	return makeValenceSeq<
		Common::seq_cat_t<
			char,
			Common::seq_pop_back_t<typename A::valseq>,
			Common::seq_pop_front_t<typename B::valseq>
		>
	>(a.t * b.t);
}

template<typename T, auto ... Vs>
std::ostream & operator<<(
	std::ostream & o,
	valence<T, Vs...> const & v
) {
	return o << v.t;
}

}
