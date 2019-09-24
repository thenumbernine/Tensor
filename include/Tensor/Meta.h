#pragma once

#include "Common/Meta.h"

namespace Tensor {

//assignment binary operations
template<typename DstType, typename SrcType> struct Assign { static void exec(DstType& dst, const SrcType& src) { dst = src; } };
template<typename DstType, typename SrcType> struct AddInto { static void exec(DstType& dst, const SrcType& src) { dst += src; } };
template<typename DstType, typename SrcType> struct SubInto { static void exec(DstType& dst, const SrcType& src) { dst -= src; } };
template<typename DstType, typename SrcType> struct MulInto { static void exec(DstType& dst, const SrcType& src) { dst *= src; } };
template<typename DstType, typename SrcType> struct DivInto { static void exec(DstType& dst, const SrcType& src) { dst /= src; } };
template<typename DstType, typename SrcType> struct AndInto { static void exec(DstType& dst, const SrcType& src) { dst &= src; } };
template<typename DstType, typename SrcType> struct OrInto { static void exec(DstType& dst, const SrcType& src) { dst |= src; } };
template<typename DstType, typename SrcType> struct XOrInto { static void exec(DstType& dst, const SrcType& src) { dst ^= src; } };

//bool specializations
template<> struct AndInto<bool, bool> { static void exec(bool& dst, const bool& src) { dst = dst && src; } };
template<> struct OrInto<bool, bool> { static void exec(bool& dst, const bool& src) { dst = dst || src; } };

//unary operations
template<typename Type> struct Identity { static const Type &exec(const Type& a) { return a; } };
template<typename Type> struct UnaryMinus { static Type exec(const Type& a) { return -a; } };

template<typename C, typename A, typename B> struct Add { static C exec(const A& a, const B& b) { return a + b; } };
template<typename C, typename A, typename B> struct Sub { static C exec(const A& a, const B& b) { return a - b; } };
template<typename C, typename A, typename B> struct Mul { static C exec(const A& a, const B& b) { return a * b; } };
template<typename C, typename A, typename B> struct Div { static C exec(const A& a, const B& b) { return a / b; } };
template<typename C, typename A, typename B> struct Equals { static C exec(const A& a, const B& b) { return a == b; } };
template<typename C, typename A, typename B> struct NotEquals { static C exec(const A& a, const B& b) { return a != b; } };
template<typename C, typename A, typename B> struct GreaterThan { static C exec(const A& a, const B& b) { return a > b; } };
template<typename C, typename A, typename B> struct GreaterThanOrEquals { static C exec(const A& a, const B& b) { return a >= b; } };
template<typename C, typename A, typename B> struct LessThan { static C exec(const A& a, const B& b) { return a < b; } };
template<typename C, typename A, typename B> struct LessThanOrEquals { static C exec(const A& a, const B& b) { return a <= b; } };
struct LogicalAnd { static bool exec(bool a, bool b) { return a && b; } };
struct LogicalOr { static bool exec(bool a, bool b) { return a || b; } };

//functor wrapper for static function
template<typename Type> struct Clamp { static Type exec(const Type& a, const Type& b, const Type& c) { return clamp(a,b,c); } };

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
	typedef const Type& Arg;
	template<int index> 
	struct Inner {
		static const Type& exec(const Type& src) { return src; }
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
	typedef const Type* Arg;
	template<int index> 
	struct Inner {
		static const Type& exec(const Type* src) { return src[index]; }
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
