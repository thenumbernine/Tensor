## Tensor Library
- compressed symmetric and antisymmetric storage
- lots of compile time and template driven stuff
- C++20
- some sort of GLSL half-compatability, though creative freedom where I want it.

## Reference:

`_vec<type, dim>` = vectors:
- `operator + - * /` scalar/vector, vector/scalar, and per-element vector/vector operations.  Including vector/vector multiply for GLSL compat, though I might change this.

`_mat<type, dim1, dim2>` = `_vec<_vec<type,dim2>,dim1>` = matrices:
- `operator + - /` scalar/matrix, matrix/scalar, and per-element matrix/matrix operations.
- `vector * matrix` as row-multplication, `matrix * vector` as column-multiplication, and `matrix * matrix` as matrix-multiplication.  Once again, GLSL compat.
- `matrixCompMult(a,b)` for component-wise multiplying two tensors.

`_sym<type, dim>` = symmetric matrices:
- `.xx .xy .xz .yy .yz .zz`

- Tensors (which are just typedef'd vectors-of-vectors-of-...)
- `_tensor<type, dim1, ..., dimN>` = construct a rank-N tensor, equivalent to nested `vec< ..., dimI>`.
- `_tensori<type, index_vec<dim1>, ..., index_sym<dimI>, ...>` = construct a tensor with specific indexes vector storage and specific pairs of indexes symmetric storage.
- `::Scalar` = get the scalar type used for this tensor.
- `::Inner`` = the next most nested vector/matrix/symmetric.
- `::rank` = for determining the tensor rank.  Vectors have rank-1, Matrices (including symmetric) have rank-2.
- `::dim< i >` = get the i'th dimension size , where i is from 0 to rank-1.
- `::localDim` = get the dimension size of the current class, equal to `dim<0>`.
- `::numNestings` = get the number of nested classes.
- `::count< i >` = get the storage size of the i'th nested class.
- `::localCount` = get the storage size of the current class, equal to `count<0>`.


Constructors:
- `()` = initialize elements to {}, aka 0 for numeric types.
- `(s)` = initialize with all elements set to `s`.
- `(x, y, z, w)` for dimensions 1-4, initialize with scalar values.
- `(function<Scalar(int i1, ...)>)` = initialize with a lambda that accepts the index of the matrix as a list of int arguments and returns the value for that index.
- `(function<Scalar(intN i)>)` = initialize with a lambda, same as above except the index is stored as an int-vector in `i`.

Builtin Indexing / Storage / Unions:
- `.s[]` element/pointer access.
- for 1D through 4D: `.x .y .z .w`, `.s0 .s1 .s2 .s2` storage.
- Right now indexing is row-major, so matrices appear as they appear in C, and so that matrix indexing `A.i.j` matches math indexing `A_ij`.  This breaks GLSL compatability.
- `.subset<size>(index), .subset<size,index>()` = return a vector reference to a subset of this vector.

Overloaded Indexing
- `(int i1, ...)` = dereference based on a list of ints.  Math `a\_ij` = `a.s[i].s[j]` in code.
- `(intN i)` = dereference based on a vector-of-ints. Same convention as above.
- `\[i1\]\[i2\]\[...\]` = dereference based on a sequence of bracket indexes.  Same convention as above.  
	Mind you that in the case of symmetric storage being used this means the [][] indexing __DOES NOT MATCH__ the .s[].s[] indexing.
	In the case of symmetric storage, for intermediate-indexes, a wrapper object will be returned.

Iterating
- `.begin() / .end() / .cbegin() / .cend()` for iterating over indexes (including duplicate elements in symmetric matrices).
- `.write().begin(), ...` for iterating over the stored elements (excluding duplicates for symmetric indexes).

Swizzle:
- `.xx() .xy() ... .wx() .ww()` 2D, `.xxx() ... .www()` 3D, `.xxxx() ... .wwww()` 4D, will return a vector-of-references.

functions:
- dot(a)
- lenSq(a)
- length(a)
- normalize(a)
- distance(a,b)
- cross(a,b)
- outer(a,b)
- determinant(m)
- inverse(m)

Depends on the "Common" project, for Exception, template metaprograms, etc.

TODO:
- get rid fo the Grid class.  
	The difference between Grid and Tensor is allocation: Grid uses dynamic allocation, Tensor uses static allocation.
	Intead, make the allocator of each dimension a templated parameter: dynamic vs static.
	This will give dynamically-sized tensors all the operations of the Tensor template class without having to re-implement them all for Grid.
	This will allow for flexible allocations: a degree-2 tensor can have one dimension statically allocated and one dimension dynmamically allocated
