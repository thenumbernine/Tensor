#pragma once

#include "Common/Meta.h"


namespace Tensor {

// TODO put in Common/Meta.h
template<typename T, typename... Args>
concept is_all_v =
	sizeof...(Args) > 0
	&& std::is_same_v<std::tuple<Args...>, Common::tuple_rep_t<T, sizeof...(Args)>>;

/*
Detects if a class is a "tensor".
These include _vec _sym _asym and subclasses (like _quat).
It's defined in the class in the TENSOR_HEADER, as a static constexpr field.

TODO should this decay_t T or should I rely on the invoker to is_tensor_v<decay_t<T>> ?
*/
template<typename T>
constexpr bool is_tensor_v = requires(T const & t) { &T::isTensorFlag; };

//https://stackoverflow.com/a/61040973
// TODO this doesn't seem to work as often as I'd like, so an easier method is just give whatever template a base class and then test is_base_v
namespace {
    template <typename, template <typename...> typename>
    struct is_instance_impl : public std::false_type {};

    template <template <typename...> typename U, typename...Ts>
    struct is_instance_impl<U<Ts...>, U> : public std::true_type {};
}
template <typename T, template <typename ...> typename U>
using is_instance = is_instance_impl<std::decay_t<T>, U>;
template <typename T, template <typename ...> typename U>
constexpr bool is_instance_v = is_instance<T, U>::value;


// I was using optional<> to capture types without instanciating them
// but optional can't wrap references, so ...
// TODO might move this somewhere like Common/Meta.h
template<typename T>
struct TypeWrapper {
	using type = T;
};

//https://codereview.stackexchange.com/a/208195
template<typename U>
struct constness_of { 
	template <typename T>
	using apply_to_t = typename std::conditional_t<
		std::is_const_v<U>,
		typename std::add_const_t<T>,
		typename std::remove_const_t<T>
	>;
};

//https://stackoverflow.com/a/48840842
// begin static_for

// std::size is supported from C++17
template <typename T, size_t N>
constexpr size_t static_size(const T (&)[N]) noexcept {
	return N;
}

template <typename ...T>
constexpr size_t static_size(const std::tuple<T...> &) {
	return std::tuple_size<std::tuple<T...> >::value;
}

template<typename Functor>
void runtime_for_lt(Functor && function, size_t from, size_t to) {
	if (from < to) {
		function(from);
		runtime_for_lt(std::forward<Functor>(function), from + 1, to);
	}
}

template <template <typename T_> class Functor, typename T>
void runtime_foreach(T & container) {
	runtime_for_lt(Functor<T>{ container }, 0, static_size(container));
}

template <typename Functor, typename T>
void runtime_foreach(T & container, Functor && functor) {
	runtime_for_lt(functor, 0, static_size(container));
}

template <typename T>
void static_consume(std::initializer_list<T>) {}

template<typename Functor, std::size_t... S>
constexpr void static_foreach_seq(Functor && function, std::index_sequence<S...>) {
	return static_consume({ (function(std::integral_constant<std::size_t, S>{}), 0)... });
}

template<std::size_t Size, typename Functor>
constexpr void static_foreach(Functor && functor) {
	return static_foreach_seq(std::forward<Functor>(functor), std::make_index_sequence<Size>());
}

// end static_for

// TODO move this to Common/Sequence.h (and take some from Common/Meta.h too)
// begin https://codereview.stackexchange.com/a/64702/265778

namespace details {
	template<typename Int, typename, Int Begin, bool Increasing>
	struct integer_range_impl;

	template<typename Int, Int... N, Int Begin>
	struct integer_range_impl<Int, std::integer_sequence<Int, N...>, Begin, true> {
		using type = std::integer_sequence<Int, N+Begin...>;
	};

	template<typename Int, Int... N, Int Begin>
	struct integer_range_impl<Int, std::integer_sequence<Int, N...>, Begin, false> {
		using type = std::integer_sequence<Int, Begin-N...>;
	};
}

template<typename Int, Int Begin, Int End>
using make_integer_range = typename details::integer_range_impl<
	Int,
	std::make_integer_sequence<Int, (Begin<End) ? End-Begin : Begin-End>,
	Begin,
	(Begin < End)
>::type;

template<std::size_t Begin, std::size_t End>
using make_index_range = make_integer_range<std::size_t, Begin, End>;

// end https://codereview.stackexchange.com/a/64702/265778



// TODO begin the implementation of wrappers ... i think i was starting to make index-operations by wrapping operations and building expression trees ...

//assignment binary operations
template<typename DstType, typename SrcType> struct Assign { static void exec(DstType& dst, SrcType const& src) { dst = src; } };
template<typename DstType, typename SrcType> struct AddInto { static void exec(DstType& dst, SrcType const& src) { dst += src; } };
template<typename DstType, typename SrcType> struct SubInto { static void exec(DstType& dst, SrcType const& src) { dst -= src; } };
template<typename DstType, typename SrcType> struct MulInto { static void exec(DstType& dst, SrcType const& src) { dst *= src; } };
template<typename DstType, typename SrcType> struct DivInto { static void exec(DstType& dst, SrcType const& src) { dst /= src; } };
template<typename DstType, typename SrcType> struct AndInto { static void exec(DstType& dst, SrcType const& src) { dst &= src; } };
template<typename DstType, typename SrcType> struct OrInto { static void exec(DstType& dst, SrcType const& src) { dst |= src; } };
template<typename DstType, typename SrcType> struct XOrInto { static void exec(DstType& dst, SrcType const& src) { dst ^= src; } };

//bool specializations
template<> struct AndInto<bool, bool> { static void exec(bool& dst, bool const& src) { dst = dst && src; } };
template<> struct OrInto<bool, bool> { static void exec(bool& dst, bool const& src) { dst = dst || src; } };

//unary operations
template<typename Type> struct Identity { static Type const &exec(Type const& a) { return a; } };
template<typename Type> struct UnaryMinus { static Type exec(Type const& a) { return -a; } };

template<typename C, typename A, typename B> struct Add { static C exec(A const& a, B const& b) { return a + b; } };
template<typename C, typename A, typename B> struct Sub { static C exec(A const& a, B const& b) { return a - b; } };
template<typename C, typename A, typename B> struct Mul { static C exec(A const& a, B const& b) { return a * b; } };
template<typename C, typename A, typename B> struct Div { static C exec(A const& a, B const& b) { return a / b; } };
template<typename C, typename A, typename B> struct Equals { static C exec(A const& a, B const& b) { return a == b; } };
template<typename C, typename A, typename B> struct NotEquals { static C exec(A const& a, B const& b) { return a != b; } };
template<typename C, typename A, typename B> struct GreaterThan { static C exec(A const& a, B const& b) { return a > b; } };
template<typename C, typename A, typename B> struct GreaterThanOrEquals { static C exec(A const& a, B const& b) { return a >= b; } };
template<typename C, typename A, typename B> struct LessThan { static C exec(A const& a, B const& b) { return a < b; } };
template<typename C, typename A, typename B> struct LessThanOrEquals { static C exec(A const& a, B const& b) { return a <= b; } };
struct LogicalAnd { static bool exec(bool a, bool b) { return a && b; } };
struct LogicalOr { static bool exec(bool a, bool b) { return a || b; } };

//functor wrapper for static function
template<typename Type> struct Clamp { static Type exec(Type const& a, Type const& b, Type const& c) { return clamp(a,b,c); } };

//accessing indexes

template<typename Type> 
struct ScalarAccess { 
	typedef Type& Arg;
	template<int index>
	struct Inner {
		static Type& exec(Type& src) { return src; }
	};
};

template<typename Type> 
struct ConstScalarAccess { 
	typedef Type const& Arg;
	template<int index> 
	struct Inner {
		static Type const& exec(Type const& src) { return src; }
	};
};

template<typename Type> 
struct ArrayAccess { 
	typedef Type* Arg;
	template<int index> 
	struct Inner {
		static Type& exec(Type* src) { return src[index]; }
	};
};

template<typename Type>
struct ConstArrayAccess { 
	typedef Type const* Arg;
	template<int index> 
	struct Inner {
		static Type const& exec(Type const* src) { return src[index]; }
	};
};

//various operations
	
template<typename AssignOp, typename Func, typename AccessDest, typename AccessA>
struct UnaryOp {
	template<int index>
	struct Inner {
		static bool exec(typename AccessDest::Arg dst, typename AccessA::Arg src) {
			AssignOp::exec(
				AccessDest::template Inner<index>::exec(dst), 
				Func::exec(
					AccessA::template Inner<index>::exec(src)
				)
			);
			return false;
		}
	};
};

template<typename AssignOp, typename Func, typename AccessDest, typename AccessA, typename AccessB>
struct BinaryOp {
	template<int index>
	struct Inner {
		static bool exec(typename AccessDest::Arg dst, typename AccessA::Arg a, typename AccessB::Arg b) {
			AssignOp::exec(
				AccessDest::template Inner<index>::exec(dst), 
				Func::exec(
					AccessA::template Inner<index>::exec(a), 
					AccessB::template Inner<index>::exec(b)
				)
			);
			return false;
		}
	};
};

template<typename AssignOp, typename Func, typename AccessDest, typename AccessA, typename AccessB, typename AccessC>
struct TernaryOp {
	template<int index>
	struct Inner {
		static bool exec(typename AccessDest::Arg dst, typename AccessA::Arg a, typename AccessB::Arg b, typename AccessC::Arg c) {
			AssignOp::exec(
				AccessDest::template Inner<index>::exec(dst),
				Func::exec(
					AccessA::template Inner<index>::exec(a),
					AccessB::template Inner<index>::exec(b),
					AccessC::template Inner<index>::exec(c)
				)
			);
			return false;
		}
	};
};

}
