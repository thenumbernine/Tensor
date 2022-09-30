#pragma once

#include "Tensor/Vector.h"

namespace Tensor {

/* tensor construction

ex: _tensor<T, dim1, ..., dimN>

equivalent to: _tensor<T, index_vec<dim>, index_vec<dim2>, ..., index_vec<dimN>>

but what about symmetric nestings?

_tensor<T, index_sym<dim1>, ..., dimN>

*/
#if 1

template<int dim>
struct index_vec {
	template<typename T>
	using type = _vec<T,dim>;
	// so (hopefully) index_vec<dim><T> == _vec<dim,T>
};

template<int dim>
struct index_sym {
	template<typename T>
	using type = _sym<T,dim>;
};

// tensor which allows custom nested storage, such as symmetric indexes

template<typename T, typename Index, typename... Indexes>
struct NestedTensorI;

template<typename T, typename Index>
struct NestedTensorI<T, Index> {
	using tensor = typename Index::template type<T>;
};

template<typename T, typename Index, typename... Indexes>
struct NestedTensorI {
	using tensor = typename Index::template type<typename NestedTensorI<T, Indexes...>::tensor>;
};

template<typename T, typename Index, typename... Indexes>
using _tensori = typename NestedTensorI<T, Index, Indexes...>::tensor;

// naive _tensor based only on scalar type and dimensions

template<typename T, int dim, int... dims>
struct NestedTensor;

template<typename T, int dim>
struct NestedTensor<T, dim> {
	using tensor = _vec<T,dim>;
};

template<typename T, int dim, int... dims>
struct NestedTensor {
	using tensor = _vec<typename NestedTensor<T, dims...>::tensor, dim>;
};

template<typename T, int dim, int... dims>
using _tensor = typename NestedTensor<T, dim, dims...>::tensor;

#else

template<typename T, int dim, int... dims>
using _tensor;

template<typename T, int dim>
using _tensor<T, dim> = _vec<T, dim>;

template<typename T, int dim, int... dims>
using _tensor = _vec<_tensor<T, dims...>, dim>;

#endif

}
