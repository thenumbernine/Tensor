#pragma once

//common metaprograms

template<int index, int end, typename Op>
struct ForLoop {
	//compile-time for-loop callback function
	//return 'true' to break the loop
	//with loop unrolling already built into compilers, 
	// this for-loop metaprogram probably is slower than an ordinary for-loop 
	static bool exec(typename Op::Input &input) {
		typedef typename Op::template Exec<index> Exec;
		if (Exec::exec(input)) return true;
		return ForLoop<index+1,end,Op>::exec(input);
	}
};

template<int end, typename Op>
struct ForLoop<end, end, Op> {
	static bool exec(typename Op::Input &input) {
		return false;
	}
};

template<bool cond, typename A, typename B>
struct If;

template<typename A, typename B>
struct If<true, A, B> {
	typedef A Type;
};

template<typename A, typename B>
struct If<false, A, B> {
	typedef B Type;
};

