#pragma once

#include <utility>	//integer_sequence
#include "Common/Sequence.h"

/*
Valence is a thin wrapper around tensor<>
except that when indexing it, it produces a valenceIndex thin wrapper around Index

should I allow valence+non-valence-tensors ?
if so then why not just consider a ValenceWrapper to be a tensor?
and then within each already-existing method I could have a test for "if it's not a valence *OR* make sure the valences match
actualy I can deny valence+nonvalence operator mixing anyways and still use thet treat-valencewrapperas-as-tensors trick
TODO think about this.

*/

namespace Tensor {

template<typename Tensor_, char ... Vs>
requires is_tensor_v<Tensor_>
struct ValenceWrapper;

template<typename T>
concept is_valence_v = T::isValenceFlag;

//Scalar & ValenceWrapper
template<typename A, typename B>
requires (is_valence_v<B> && !is_valence_v<A> && !is_tensor_v<A>)
auto inner(A const & a, B const & b);
	
//ValenceWrapper & Scalar
template<typename A, typename B>
requires (is_valence_v<A> && !is_valence_v<B> && !is_tensor_v<B>)
auto inner(A const & a, B const & b);

//ValenceWrapper & ValenceWrapper
template<typename A, typename B>
requires (
	is_valence_v<A> &&
	is_valence_v<B> &&
	std::is_same_v<typename A::Tensor::dimseq, typename B::Tensor::dimseq>
)
auto inner(A const & a, B const & b);

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

template<typename Tensor_, char ... Vs>
requires is_tensor_v<Tensor_>
struct ValenceWrapper {
	using This = ValenceWrapper;
	using Tensor = Tensor_;
	static constexpr bool isValenceFlag = true;
	using Scalar = typename Tensor::Scalar;
	static constexpr int rank = Tensor::rank;
	
	template<typename NewTensor>
	using ReplaceTensor = ValenceWrapper<NewTensor, Vs...>;

	Tensor t;
	ValenceWrapper() {}
	ValenceWrapper(Tensor const & t_) : t(t_) {}
	ValenceWrapper(Tensor && t_) : t(t_) {}

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
		return valenceSeq<Common::seq_pop_front_t<valseq>>(t(i));
	}
	template<typename Int>\
	requires std::is_integral_v<Int>\
	constexpr decltype(auto) operator()(Int const i) const {
		return valenceSeq<Common::seq_pop_front_t<valseq>>(t(i));
	}

	//operator[](Int), operator()(Int...) fwd to operator()(Int)(...)
	TENSOR_ADD_RANK1_CALL_INDEX_AUX()
	//operator()(vec<Int,N>) fwd to operator()(Int...)
	TENSOR_ADD_INT_VEC_CALL_INDEX()

	// TODO Index based operator() and a whole set of wrappers of Index expression operators...

	
	//TENSOR_ADD_MATH_MEMBER_FUNCS()
	// TODO member method forwarding ...?
	// or just rely on * -> operators?
	// nahhh, because those won't check and apply valence wrappers.

	// c_I = a_I * b_I, no change in valence
	auto elemMul(This const & o) const {
		return This(Tensor::elemMul(t, o.t));
	}
	auto matrixCompMult(This const & o) const {
		return This(Tensor::matrixCompMult(t, o.t));
	}
	auto hadamard(This const & o) const {
		return This(Tensor::hadamard(t, o.t));
	}

	template<typename B>
	auto inner(B const & o) const {
		return Tensor::inner(*this, o);
	}
};


template<typename Tensor, typename Seq>
struct valenceSeqImpl;
template<typename Tensor, typename I, I i1, I... is>
struct valenceSeqImpl<Tensor, std::integer_sequence<I, i1, is...>> {
	using type = ValenceWrapper<Tensor, i1, is...>;
};
template<typename Tensor, typename Seq>
using ValenceWrapperSeq = typename valenceSeqImpl<Tensor, Seq>::type;

// so you don't have to explicitly write the first template arg ...
template<char ... vs>
auto valence(auto const & v) {
	return ValenceWrapper<std::decay_t<decltype(v)>, vs...>(v);
}
template<char ... vs>
auto valence(auto && v) {
	return ValenceWrapper<std::decay_t<decltype(v)>, vs...>(v);
}

template<typename Seq>
auto valenceSeq(auto const & v) {
	return ValenceWrapperSeq<std::decay_t<decltype(v)>, Seq>(v);
}
template<typename Seq>
auto valenceSeq(auto && v) {
	return ValenceWrapperSeq<std::decay_t<decltype(v)>, Seq>(v);
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
	return valenceSeq<typename A::valseq>(a.t op b);\
}\
template<typename A, typename B>\
requires (!is_valence_v<A> && !is_tensor_v<A> && is_valence_v<B>)\
decltype(auto) operator op(A const & a, B const & b) {\
	return valenceSeq<typename B::valseq>(a op b.t);\
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
	return valenceSeq<typename A::valseq>(a.t op b);\
}\
\
template<typename A, typename B>\
requires (\
	!is_valence_v<A> && !is_tensor_v<A> &&\
	!std::is_base_of_v<std::ios_base, std::decay_t<A>> &&\
	is_valence_v<B>\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	return valenceSeq<typename B::valseq>(a op b.t);\
}

#define TENSOR_VALENCE_VALENCE_OP(op)\
template<typename A, typename B>\
requires (\
	is_valence_v<A> && is_valence_v<B> &&\
	std::is_same_v<typename A::valseq, typename B::valseq>\
)\
decltype(auto) operator op(A const & a, B const & b) {\
	return valenceSeq<typename A::valseq>(a.t op b.t);\
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
	return valenceSeq<
		Common::seq_cat_t<
			char,
			Common::seq_pop_back_t<typename A::valseq>,
			Common::seq_pop_front_t<typename B::valseq>
		>
	>(a.t * b.t);
}

//Scalar & ValenceWrapper
template<typename A, typename B>
requires (is_valence_v<B> && !is_valence_v<A> && !is_tensor_v<A>)
auto inner(A const & a, B const & b) {
	return inner(a, b.t);
}

//ValenceWrapper & Scalar
template<typename A, typename B>
requires (is_valence_v<A> && !is_valence_v<B> && !is_tensor_v<B>)
auto inner(A const & a, B const & b) {
	return inner(a.t, b);
}

//ValenceWrapper & ValenceWrapper
template<typename A, typename B>
requires (
	is_valence_v<A> &&
	is_valence_v<B> &&
	std::is_same_v<typename A::Tensor::dimseq, typename B::Tensor::dimseq>
)
auto inner(A const & a, B const & b) {
	static_assert(
		[]<size_t...i>(std::index_sequence<i...>) constexpr -> bool {
			return (
				(Common::seq_get_v<i, typename A::valseq> != Common::seq_get_v<i, typename B::valseq>)
				&& ... && (true)
			);
		}(std::make_index_sequence<A::rank>{}),
		"valence mismatch"
	);
	return inner(a.t, b.t);
}

template<typename T, auto ... Vs>
std::ostream & operator<<(
	std::ostream & o,
	ValenceWrapper<T, Vs...> const & v
) {
	return o << v.t;
}

}
