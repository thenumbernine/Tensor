#pragma once

//from https://bitbucket.org/rdpate/kniht

template<class D, class B> 
D& crtp_cast(B& p) {
	return static_cast<D&>(p);
}

template<class D, class B>
D const& crtp_cast(B const& p) {
	return static_cast<D const&>(p);
}

template<class D, class B>
D volatile& crtp_cast(B volatile& p) {
	return static_cast<D volatile&>(p);
}

template<class D, class B>
D const volatile& crtp_cast(B const volatile& p) {
	return static_cast<D const volatile&>(p);
}

