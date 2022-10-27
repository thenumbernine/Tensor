#include "Tensor/Vector.h"
#include "Common/Macros.h"
#include <functional>

namespace Tensor {

/*
partial derivative index operator
(partial derivative alone one coordinate)

finite difference coefficients for center-space finite-difference partial derivatives found at
http://en.wikipedia.org/wiki/Finite_difference_coefficients
*/

template<typename Real, int order>
struct PartialDerivativeCoeffs;

template<typename Real>
struct PartialDerivativeCoeffs<Real, 2> {
	static constexpr std::array<Real, 1> coeffs = { 1./2. };
};

template<typename Real>
struct PartialDerivativeCoeffs<Real, 4> {
	static constexpr std::array<Real, 2> coeffs = { 2./3., -1./12. };
};

template<typename Real>
struct PartialDerivativeCoeffs<Real, 6> {
	static constexpr std::array<Real, 3> coeffs = { 3./4., -3./20., 1./60. };
};

template<typename Real>
struct PartialDerivativeCoeffs<Real, 8> {
	static constexpr std::array<Real, 4> coeffs = { 4./5., -1./5., 4./105., -1./280. };
};

// continuous derivative

template<int order = 2>
auto partialDerivative(
	auto f,
	auto x,
	typename decltype(f(x))::Scalar h = .01
) {
	using T = decltype(f(x));
	using S = typename T::Scalar;
	using C = PartialDerivativeCoeffs<S, order>;
	static_assert(T::isSquare);		// all dimensions match
	constexpr int dim = T::template dim<0>;
	auto xofs = [](
		auto x,
		S h,
		int k,			// which dimension
		S offset		// how far
	) {
		auto x2 = x;
		x2[k] += h * offset;
		return x2;
	};
	_vec<T, dim> result; 			//first index is derivative
	for (int k = 0; k < dim; ++k) {
		
		result[k] = [&]<int ... i>(std::integer_sequence<int, i...>) constexpr -> T {
			return (((
				f(xofs(x,h,k,i+1))
				- f(xofs(x,h,k,-i-1))
			) * C::coeffs[i] / h) + ... + (0));
		}(Common::make_integer_range<int, 0, C::coeffs.size()>{});
	}
	return result;
}

// grid derivatives
// TODO redo the whole Grid class

template<typename Type, int rank> 
Type getOffset(
	std::function<Type(intN<rank>)> f,
	intN<rank> index,
	int dim,
	int offset)
{
	index(dim) += offset;
	return f(index);
}

/*
partial derivative operator
for now let's use 2-point everywhere: d/dx^i f(x) ~= (f(x + dx^i) - f(x - dx^i)) / (2 * |dx^i|)
	index = index in grid of cell to pull the specified field
	k = dimension to differentiate across
*/
template<int order, typename Real, int dim, typename InputType>
struct PartialDerivativeGridImpl;

template<int order, typename Real, int dim, typename InputType>
requires is_tensor_v<InputType> && std::is_same_v<Real, typename InputType::Scalar>
struct PartialDerivativeGridImpl<order, Real, dim, InputType> {
	static constexpr auto rank = InputType::rank;
	static constexpr decltype(auto) exec(
		intN<dim> const & gridIndex,
		_vec<Real, dim> const & dx,
		std::function<InputType(intN<dim> index)> f
	) {
		using Coeffs = PartialDerivativeCoeffs<Real, order>;
		return _vec<InputType, dim>([&](intN<rank+1> dstIndex) -> Real {
			int gradIndex = dstIndex(0);
			intN<rank> srcIndex;
#if 0	// sequences, return types, and release mode gcc, give me a warning about using this trick:			
			[&]<int ... i>(std::integer_sequence<int, i...>) constexpr -> int {
				return ((srcIndex(i) = dstIndex(i+1)), ..., (0));
			}(std::make_integer_sequence<int, rank>{});
#else
			for (int i = 0; i < rank; ++i) {
				srcIndex(i) = dstIndex(i+1);
			}
#endif
			return [&]<int ... i>(std::integer_sequence<int, i...>) constexpr -> Real {
				return (((
					getOffset<InputType, dim>(f, gridIndex, gradIndex, i)(srcIndex) 
					- getOffset<InputType, dim>(f, gridIndex, gradIndex, -i)(srcIndex)
				) * Coeffs::coeffs[i-1]) + ... + (0)) / dx(gradIndex);
			}(Common::make_integer_range<int, 1, Coeffs::coeffs.size()>{});
		});
	}
};

template<int order, typename Real, int dim>
struct PartialDerivativeGridImpl<order, Real, dim, Real> {
	using InputType = Real;
	static constexpr decltype(auto) exec(
		intN<dim> const &gridIndex,
		_vec<Real, dim> const &dx, 
		std::function<InputType(intN<dim> index)> f
	) {
		using Coeffs = PartialDerivativeCoeffs<Real, order>;
		return _vec<Real, dim>([&](int gradIndex) -> Real {
			return [&]<int ... i>(std::integer_sequence<int, i...>) constexpr -> Real {
				return (((
					getOffset<InputType, dim>(f, gridIndex, gradIndex, i)
					- getOffset<InputType, dim>(f, gridIndex, gradIndex, -i)
				) * Coeffs::coeffs[i-1]) + ... + (0)) / dx(gradIndex);
			}(Common::make_integer_range<int, 1, Coeffs::coeffs.size()>{});
		});
	}
};

template<int order, typename Real, int dim, typename InputType>
auto partialDerivativeGrid(
	intN<dim> const & index,
	_vec<Real, dim> const & dx,
	std::function<InputType(intN<dim>)> f
) {
	return PartialDerivativeGridImpl<order, Real, dim, InputType>::exec(index, dx, f);
}

}
