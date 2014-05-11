#pragma once

//common metaprograms

template<int index, int end, typename Op>
struct ForLoop {
	static void exec(typename Op::Input &input) {
		typedef typename Op::template Exec<index> Exec;
		Exec::exec(input);
		ForLoop<index+1,end,Op>::exec(input);
	}
};

template<int end, typename Op>
struct ForLoop<end, end, Op> {
	static void exec(typename Op::Input &input) {
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

