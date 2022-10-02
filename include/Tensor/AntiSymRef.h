#pragma once

// wrapper of a ref of a type that reads and writes the negative of the value its wrapping

#include <functional>	//std::reference_wrapper
#include <optional>
#include <ostream>		//std::ostream

namespace Tensor {

/*
use the 'rank' field to check and see if we're in a _vec (or a _sym) or not
 TODO use something more specific to this file in case other classes elsewhere use 'rank'
*/
template<typename T>
constexpr bool is_tensor_v = requires(T const & t) { T::rank; };



//https://stackoverflow.com/a/61040973
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

typedef enum {
	POSITIVE, // == 0 // so that how1 ^ how2 produces the sign for AntiSymRef1-of-AntiSymRef2
	NEGATIVE, // == 1
	ZERO,
} AntiSymRefHow;

template<typename T>
struct AntiSymRef {
	using This = AntiSymRef;
	using Type = T;
	
	static_assert(!is_instance_v<T, AntiSymRef>);	//never wrap AntiSymRef<AntiSymRef<T>>, instead just use 1 wrapper, flip 'how' if necessary.

	std::optional<std::reference_wrapper<T>> x;

	AntiSymRefHow how = ZERO;
	
	AntiSymRef() {}
	AntiSymRef(std::reference_wrapper<T> const & x_, AntiSymRefHow how_) : x(x_), how(how_) {}
	AntiSymRef(AntiSymRef const & r) : x(r.x), how(x.how) {}
	
	AntiSymRef(AntiSymRef && r)
	: x(r.x), how(x.how) {}
	
	This & operator=(T const & y) {
		if (how == AntiSymRefHow::POSITIVE) {
			(*x).get() = y;
		} else if (how == AntiSymRefHow::NEGATIVE) {
			(*x).get() = -y;
		} else {//if (how == ZERO) {
			//ZERO should only be wrapping temp elements outside the antisymmetric matrix
		}
		return *this;
	}
	
	This & operator=(T && y) {
		if (how == AntiSymRefHow::POSITIVE) {
			(*x).get() = y;
		} else if (how == AntiSymRefHow::NEGATIVE) {
			(*x).get() = -y;
		} else {//if (how == ZERO) {
			// no write
		}
		return *this;
	}

	operator T() const {
		if (how == AntiSymRefHow::POSITIVE) {
			return (*x).get();
		} else if (how == AntiSymRefHow::NEGATIVE) {
			return -(*x).get();
		} else {//if (how == ZERO) {
			return {};
		}
	}

	template<typename... Args>
	auto operator()(Args&&... args) const
	//requires (std::is_invocable_v<T()>) 
	requires (is_tensor_v<T>)
	{
		using R = decltype((*x).get()(std::forward<Args>(args)...));
		if constexpr (is_instance_v<R, AntiSymRef>) {
			using RT = typename R::Type;	// TODO nested-most?  verified for two-nestings deep, what about 3?
			if (how != AntiSymRefHow::POSITIVE && how != AntiSymRefHow::NEGATIVE) {
				return AntiSymRef<RT>();
			} else {
				R && r = (*x).get()(std::forward<Args>(args)...);
				if (r.how != AntiSymRefHow::POSITIVE && r.how != AntiSymRefHow::NEGATIVE) {
					return AntiSymRef<RT>();
				} else {
					return AntiSymRef<RT>(*r.x, (AntiSymRefHow)((int)how ^ (int)r.how));
				}
			}
		} else {
			if (how != AntiSymRefHow::POSITIVE && how != AntiSymRefHow::NEGATIVE) {
				return AntiSymRef<R>();
			} else {
				R && r = (*x).get()(std::forward<Args>(args)...);
				return AntiSymRef<R>(std::ref(r), how);
			}
		}
	}
};

static_assert(is_instance_v<AntiSymRef<double>, AntiSymRef>);
static_assert(is_instance_v<AntiSymRef<AntiSymRef<double>>, AntiSymRef>);

template<typename T>
bool operator==(AntiSymRef<T> const & a, T const & b) {
	return a.operator T() == b;
}

template<typename T>
bool operator==(T const & a, AntiSymRef<T> const & b) {
	return a == b.operator T();
}

template<typename T>
bool operator==(AntiSymRef<T> const & a, AntiSymRef<T> const & b) {
	return a.operator T() == b.operator T();
}

template<typename T> bool operator!=(AntiSymRef<T> const & a, T const & b) { return !operator==(a,b); }
template<typename T> bool operator!=(T const & a, AntiSymRef<T> const & b) { return !operator==(a,b); }
template<typename T> bool operator!=(AntiSymRef<T> const & a, AntiSymRef<T> const & b) { return !operator==(a,b); }

template<typename T>
std::ostream & operator<<(std::ostream & o, AntiSymRef<T> const & t) {
	return o << t.operator T();
}

}
