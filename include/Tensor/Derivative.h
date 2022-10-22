#include "Tensor/Vector.h"
#include "Common/Macros.h"
#include <functional>

namespace Tensor {

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
partial derivative index operator
(partial derivative alone one coordinate)

finite difference coefficients for center-space finite-difference partial derivatives found at
http://en.wikipedia.org/wiki/Finite_difference_coefficients
*/

template<typename Real, int order>
struct PartialDerivCoeffs;

template<typename Real>
struct PartialDerivCoeffs<Real, 2> {
	static constexpr std::array<Real, 1> coeffs = { 1./2. };
};

template<typename Real>
struct PartialDerivCoeffs<Real, 4> {
	static constexpr std::array<Real, 2> coeffs = { 2./3., -1./12. };
};

template<typename Real>
struct PartialDerivCoeffs<Real, 6> {
	static constexpr std::array<Real, 3> coeffs = { 3./4., -3./20., 1./60. };
};

template<typename Real>
struct PartialDerivCoeffs<Real, 8> {
	static constexpr std::array<Real, 4> coeffs = { 4./5., -1./5., 4./105., -1./280. };
};

/*
partial derivative operator
for now let's use 2-point everywhere: d/dx^i f(x) ~= (f(x + dx^i) - f(x - dx^i)) / (2 * |dx^i|)
	index = index in grid of cell to pull the specified field
	k = dimension to differentiate across
*/
template<int order, typename Real, int dim, typename InputType>
struct PartialDerivativeClass;

template<int order, typename Real, int dim, typename InputType>
requires is_tensor_v<InputType> && std::is_same_v<Real, typename InputType::Scalar>
struct PartialDerivativeClass<order, Real, dim, InputType> {
	using RealN = _vec<Real, dim>;
	using OutputType = _vec<InputType, dim>;
	using FuncType = std::function<InputType(intN<dim> index)>;
	static constexpr auto rank = InputType::rank;
	OutputType operator()(
		intN<dim> const & gridIndex,
		RealN const & dx,
		FuncType f)
	{
		using Coeffs = PartialDerivCoeffs<Real, order>;
		return OutputType([&](intN<rank+1> dstIndex) -> Real {
			int gradIndex = dstIndex(0);
			intN<rank> srcIndex;
			[&]<size_t ... i>(std::index_sequence<i...>) constexpr -> int {
				return ((srcIndex(i) = dstIndex(i+1)), ..., (0));
			}(std::make_index_sequence<rank>{});
			return [&]<size_t ... i>(std::index_sequence<i...>) constexpr -> Real {
				return (((
					getOffset<InputType, dim>(f, gridIndex, gradIndex, i)(srcIndex) 
					- getOffset<InputType, dim>(f, gridIndex, gradIndex, -i)(srcIndex)
				) * Coeffs::coeffs[i-1]) + ... + (0)) / dx(gradIndex);
			}(Common::make_index_range<1, Coeffs::coeffs::size()>{});
		});
	}
};

template<int order, typename Real, int dim>
struct PartialDerivativeClass<order, Real, dim, Real> {
	using RealN = _vec<Real, dim>;
	using InputType = Real;
	using OutputType = RealN;
	using FuncType = std::function<InputType(intN<dim> index)>;
	OutputType operator()(
		intN<dim> const &gridIndex,
		RealN const &dx, 
		FuncType f)
	{
		using Coeffs = PartialDerivCoeffs<Real, order>;
		return OutputType([&](int gradIndex) -> Real {
			return [&]<size_t ... i>(std::index_sequence<i...>) constexpr -> Real {
				return (((
					getOffset<InputType, dim>(f, gridIndex, gradIndex, i)
					- getOffset<InputType, dim>(f, gridIndex, gradIndex, -i)
				) * Coeffs::coeffs[i-1]) + ... + (0)) / dx(gradIndex);
			}(Common::make_index_range<1, Coeffs::coeffs::size()>{});
		});
	}
};

template<int order, typename Real, int dim, typename InputType>
auto partialDerivative(
	intN<dim> const &index,
	_vec<Real, dim> const &dx,
	typename PartialDerivativeClass<order, Real, dim, InputType>::FuncType f)
{
	return PartialDerivativeClass<order, Real, dim, InputType>()(index, dx, f);
}

}
