## Tensor Library
- Familiar vector and math support, 2D 3D 4D.
- Arbitrary-dimension, arbitrary-rank.
- Compressed symmetric and antisymmetric storage.
- Lots of compile time and template driven stuff.
- C++20
- Some sort of GLSL half-compatability, though creative freedom where I want it.

## Reference:

`_vec<type, dim>` = vectors:
- `operator + - * /` scalar/vector, vector/scalar, and per-element vector/vector operations.  Including vector/vector multiply for GLSL compat, though I might change this.

`_mat<type, dim1, dim2>` = `_vec<_vec<type,dim2>,dim1>` = matrices:
- `operator + - /` scalar/matrix, matrix/scalar, and per-element matrix/matrix operations.
- `vector * matrix` as row-multplication, `matrix * vector` as column-multiplication, and `matrix * matrix` as matrix-multiplication.  Once again, GLSL compat.
- `matrixCompMult(a,b)` for component-wise multiplying two tensors.

`_sym<type, dim>` = symmetric matrices:
- `.x_x .x_y .x_z .y_y .y_z .z_z .x_w .y_w .z_w .w_w` storage, `.y_x .z_x, .z_y` union'd access.

- Tensors (which are just typedef'd vectors-of-vectors-of-...)
- `_tensor<type, dim1, ..., dimN>` = construct a rank-N tensor, equivalent to nested `vec< ..., dimI>`.
- `_tensori<type, index_vec<dim1>, ..., index_sym<dimI>, ...>` = construct a tensor with specific indexes vector storage and specific pairs of indexes symmetric storage.
- `::Scalar` = get the scalar type used for this tensor.
- `::Inner` = the next most nested vector/matrix/symmetric.
- `::rank` = for determining the tensor rank.  Vectors have rank-1, Matrices (including symmetric) have rank-2.
- `::dim<i>` = get the i'th dimension size , where i is from 0 to rank-1.
- `::dims()` = get a int-vector with all the dimensions.
- `::localDim` = get the dimension size of the current class, equal to `dim<0>`.
- `::numNestings` = get the number of nested classes.
- `::count<i>` = get the storage size of the i'th nested class.
- `::localCount` = get the storage size of the current class, equal to `count<0>`.
- `::replaceScalar<T>` = create a type of this nesting of tensor templates, except with the scalar-most replaced by T.
- `::Nested<i>` = get the i'th nested type from our tensor type.

Constructors:
- `()` = initialize elements to {}, aka 0 for numeric types.
- `(s)` = initialize with all elements set to `s`.
- `(x, y, z, w)` for dimensions 1-4, initialize with scalar values.
- `(function<Scalar(int i1, ...)>)` = initialize with a lambda that accepts the index of the matrix as a list of int arguments and returns the value for that index.
- `(function<Scalar(intN i)>)` = initialize with a lambda, same as above except the index is stored as an int-vector in `i`.
- `(tensor t)` = initialize from another tensor.  Truncate dimensions.  Uninitialized elements are set to {}.

Builtin Indexing / Storage / Unions:
- `.s[]` element/pointer access.
- for 1D through 4D: `.x .y .z .w`, `.s0 .s1 .s2 .s2` storage.
- Right now indexing is row-major, so matrices appear as they appear in C, and so that matrix indexing `A.i.j` matches math indexing `A_ij`.  This disagrees with GL compatability, so you'll have to upload your matrices to GL transposed.
- `.subset<size>(index), .subset<size,index>()` = return a vector reference to a subset of this vector.

Overloaded Indexing
- `(int i1, ...)` = dereference based on a list of ints.  Math `a_ij` = `a.s[i].s[j]` in code.
- `(intN i)` = dereference based on a vector-of-ints. Same convention as above.
- `[i1][i2][...]` = dereference based on a sequence of bracket indexes.  Same convention as above.  
	Mind you that in the case of symmetric storage being used this means the [][] indexing __DOES NOT MATCH__ the .s[].s[] indexing.
	In the case of symmetric storage, for intermediate-indexes, a wrapper object will be returned.

Iterating
- `.begin() / .end() / .cbegin() / .cend()` = read-iterators, for iterating over indexes (including duplicate elements in symmetric matrices).
- `.write().begin(), ...` = write-iterators, for iterating over the stored elements (excluding duplicates for symmetric indexes).
- read-iterators have `.index` as the int-vector of the index into the tensor.
- write-iterators have `.readIndex` as the int-vector lookup index and `.writeIndex` as the nested-storage int-vector.

Swizzle will return a vector-of-references:
- 2D: `.xx() .xy() ... .wz() .ww()`
- 3D: `.xxx() ... .www()`
- 4D: `.xxxx() ... .wwww()` 

functions:
- `dot(a,b)` = Frobenius dot.
	$$dot(a,b) := a^I \cdot b_I$$
- `lenSq(a)` = For vectors this is the length-squared.  It is a self-dot, for vectors this is equal to the length squared, for tensors this is the Frobenius norm (... squared? Math literature mixes up the definition of "norm" between the sum-of-squares and its square-root.).
	$$lenSq(a) := |a|^2 = a^I a_I$$
- `length(a)` = For vectors this is the length.  It is the sqrt of the self-dot.
	$$length(a) := |a| = \sqrt{a^I a_I}$$
- `normalize(a)` = For vectors this returns a unit.  It is a tensor divided by its sqrt-of-self-dot
	$$normalize(a) := a / |a|$$
- `distance(a,b)` = Length of the difference of two tensors.
	$$distance(a,b) := |b - a|$$
- `cross(a,b)` = 3D vector cross product.  TODO generalize to something with Levi-Civita permutation tensor.
	$${cross(a,b)_i} := {\epsilon_{ijk}} b^j c^k$$ 
- `outer(a,b)` = Tensor outer product.  Two vectors make a matrix.  A vector and a matrix make a rank-3.  Etc.
	$$outer(a,b)_{IJ} := a_I b_J$$
- `determinant(m)` = Matrix determinant, equal to `dot(cross(m.x, m.y), m.z)`.
	$$determinant(a) := det(a) = \epsilon_I {a^{i_1}}_1 {a^{i_2}}_2 {a^{i_3}}_3 ... {a^{i_n}}_n$$
- `inverse(m)` = Matrix inverse, for rank-2 tensors.
	$${inverse(a)^{i_1}}_{j_1} := \frac{1}{(n-1)! det(a)} \delta^I_J {a^{j_2}}_{i_2} {a^{j_3}}_{i_3} ... {a^{j_n}}_{i_n}$$
- `transpose(m)` = Matrix transpose, for rank-2 tensors.
	$$transpose(a)_{ij} = a_{ji}$$
- `diagonal(m)` = Matrix diagonal from vector.
	$${diagonal(a)_{ij} = \delta_{ij} \cdot a_i$$
- `trace(m)` = Matrix trace = matrix contraction between two indexes.

## Familiar Types

Sorry GLSL, Cg wins this round:
- `floatN<N>` = N-dimensional float vector.
- `float2, float3, float4` = 1D, 2D, 3D float vector.
- `float2x2, float2x3, float2x4, float3x2, float3x3, float3x4, float4x2, float4x3, float4x4` = matrix types.
- `float2s2, float3s3, float4s4` = symmetric matrix types.
- ... same with bool, char, uchar, short, ushort, int, uint, float, double, ldouble, size, intptr, uintptr.
- `_vec2<T>, _vec3<T>, _vec4<T>` = templated fixed-size vectors.
- `_mat2x2<T>, _mat2x3<T>, _mat2x4<T>, _mat3x2<T>, _mat3x3<T>, _mat3x4<T>, _mat4x2<T>, _mat4x3<T>, _mat4x4<T>` = templated fixed-size matrices.
- `_sym2<T>, _sym3<T>, _sym4<T>` = templated fixed-size symmetric matrices.

Depends on the "Common" project, for Exception, template metaprograms, etc.

TODO:
- get rid fo the Grid class.  
	The difference between Grid and Tensor is allocation: Grid uses dynamic allocation, Tensor uses static allocation.
	Intead, make the allocator of each dimension a templated parameter: dynamic vs static.
	This will give dynamically-sized tensors all the operations of the Tensor template class without having to re-implement them all for Grid.
	This will allow for flexible allocations: a degree-2 tensor can have one dimension statically allocated and one dimension dynmamically allocated

- 1) ability to expand specific-rank of a tensor from sym (or antisym) into vec-of-vec
parallel:
- 1) how about antisym(rank-2)
	- for a n x n antisym we have only n*(n-1)/2 instead of symmetric n*(n+1)/2
	- once we have this, why not rank-3 rank-4 etc sym or antisym ... I'm sure there's some math on how to calculate the unique # of vars
	- first this starts with a negative-ref wrapper, which initializes with another variable, and always read/writes the negative of its source.
- 2) "expandStorage<>" ... for our nested tensor-of-index-of-sym-of...etc ... need a template option for taking one specific index, and - for whatever sturcuture is there (vec, sym, antisym) turning it into vecs-of-vecs 
	- then do this when you transpose() it, when you operator* it, any time you need to produce an output type based on input types whose dimensions might overlap optimized storage structures.
		- or better, make a permuteOrder() function, have this "expandStorage<>" on all its passed indexes, then have it run across the permute indexes.

- GetNestedForIndex<i> = 
	- then use this for 'dim<i> = GetNestedForIndex<i>::dim`.
	- then ExpandStorage<i> would .... ?

- more flexible exterior product (cross, wedge, determinant)
- more flexible multiplication ... it's basically outer then contract of lhs last and rhs first indexes ... though I could optimize to a outer+contract-N indexes
	- for mul(A a, B b), this would be "expandStorage" on the last of A and first of B b, then "replaceScalar" the nesting-minus-one of A with the nesting-1 of B
- index notation summation?  mind you that it shoud preserve non-summed index memory-optimization structures.
- better function matching for derivatives?
- shorthand those other longwinded GLSL names like "inverse"=>"inv", "determinant"=>"det", "transpose"=>"tr" "normalize"=>"unit"
- make transpose a specialization of permuteIndexes()
- multiply as contraction of indexes
	- for multiply and for permute, some way to extract the flags for symmetric/not on dif indexes
	  so when i produce result types with some indexes moved/removed, i'll know when to expand symmetric into vec-of-vec
- move secondderivative from Relativity to Tensor
- move covariantderivative from Relativity to Tensor
- move Hydro/Inverse.h's GaussJordan solver into Tensor/Inverse

- change sym and asym to access upper-triangular instead of lower-triangular
