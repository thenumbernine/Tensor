## Tensor Library
- Familiar vector and math support, 2D 3D 4D.
- Arbitrary-dimension, arbitrary-rank.
- Compressed symmetric and antisymmetric storage.
- Lots of compile time and template driven stuff.
- C++20
- Some sort of GLSL half-compatability, though creative freedom where I want it.

## Reference:

`_vec<type, dim>` = vectors:
- `.s[]` element/pointer access.
- for 1D through 4D: `.x .y .z .w`, `.s0 .s1 .s2 .s2` storage.
- `.subset<size>(index), .subset<size,index>()` = return a vector reference to a subset of this vector.
- `operator + - * /` scalar/vector, vector/scalar, and per-element vector/vector operations.  Including vector/vector multiply for GLSL compat, though I might change this.

`_mat<type, dim1, dim2>` = `_vec<_vec<type,dim2>,dim1>` = matrices:
- `operator + - /` scalar/matrix, matrix/scalar, and per-element matrix/matrix operations.
- `vector * matrix` as row-multplication, `matrix * vector` as column-multiplication, and `matrix * matrix` as matrix-multiplication.  Once again, GLSL compat.
- Right now indexing is row-major, so matrices appear as they appear in C, and so that matrix indexing `A.i.j` matches math indexing `A_ij`.  This disagrees with GL compatability, so you'll have to upload your matrices to GL transposed.

`_sym<type, dim>` = symmetric matrices:
- `.x_x .x_y .x_z .y_y .y_z .z_z .x_w .y_w .z_w .w_w` storage, `.y_x .z_x, .z_y` union'd access.

`_asym<type, dim>` = antisymmetric matrices:
- `.x_x() .w_w()` access methods

`_quat<type>` = quaternion.  Subclass of `_vec4<type>`.

- Tensors (which are just typedef'd vectors-of-vectors-of-...)
- `_tensor<type, dim1, ..., dimN>` = construct a rank-N tensor, equivalent to nested `vec< ..., dimI>`.
- `_tensori<type, I1, I2, I3...>` = construct a tensor with specific indexes vector storage and specific pairs of indexes symmetric storage.  `I1 I2` etc are one of the following: `index_vec<dim>` for a single index of dimension `dim`, `index_sym<dimI>` for two symmetric indexes of dimension `dim`, or `index_asym<dim>` for two antisymmetric indexes of dimension `dim`.
- `_tensorr<type, dim, rank>` = construct a tensor of rank-`rank` with all dimensions `dim`.
- `::Scalar` = get the scalar type used for this tensor.
- `::Inner` = the next most nested vector/matrix/symmetric.
- `::rank` = for determining the tensor rank.  Vectors have rank-1, Matrices (including symmetric) have rank-2.
- `::dim<i>` = get the i'th dimension size , where i is from 0 to rank-1.
- `::dims` = get a int-vector with all the dimensions.  For rank-1 this is just an int.  Maybe I'll change it to be intN for all ranks, not sure.
- `::localDim` = get the dimension size of the current class, equal to `dim<0>`.
- `::numNestings` = get the number of nested classes.
- `::count<i>` = get the storage size of the i'th nested class.  i is from 0 to numNestings-1.
- `::localCount` = get the storage size of the current class, equal to `count<0>`.
- `::Nested<i>` = get the i'th nested type from our tensor type, where i is from 0 to numNestings-1.
- `::numNestingsToIndex<i>` = gets the number of nestings deep that the index 'i' is located.
- `::InnerForIndex<i>` = get the type associated with the i'th index, where i is from 0 to rank-1.  Equivalent to `::Nested<numNestingsToIndex<i>>`  vec's 0 will point to itself, sym's and asym's 0 and 1 will point to themselves, all others drill down.
- `::ReplaceInner<T>` = replaces this tensor's Inner with a new type, T.
- `::ReplaceScalar<T>` = create a type of this nesting of tensor templates, except with the scalar-most type replaced by T.
- `::ExpandIthIndex<i>` = produce a type with only the storage at the i'th index replaced with expanded storage.  Expanded storage being a vec-of-vec-of...-vec's with nesting equal to the desired tensor rank.  So a sym's ExpandIthIndex<0> or <1> would produce a vec-of-vec.  a sym-of-vec's ExpandIthIndex<2> would return the same type, and a vec-of-sym's ExpandIthIndex<0> would return the same type.
- `::ExpandIndex<i1, i2, ...>` = expand the all indexes listed.
- `::ExpandIndexSeq<std::integer_sequence<int, i1...>>` = expands the indexes in the `integer_sequence<int, ...>`
- `::ExpandAllIndexes<>` = produce a type with all storage replaced with expanded storage.  Expanded storage being a vec-of-vec-of...-vec's with nesting equal to the desired tensor rank.  Equivalent to `T::ExpandIthIndex<0>::...::ExpandIthIndex<T::rank-1>`.  Also equivalent to an equivalent tensor with no storage optimizations, i.e. `_tensori<Scalar, dim1, ..., dimN>`.
- `::RemoveIthIndex<i>` = Removes the i'th index.  First expands the storage at that index, so that any rank-2's will turn into vecs.
- `::RemoveIndex<i1, i2, ...>` = Removes all indexes.  

Tensor Template Helpers (subject to change)
- `is_tensor_v<T>` = is it a tensor storage type?
- `is_vec_v<T>` = is it a _vec<T,N>?
- `is_sym_v<T>` = is it a _sym<T,N>?
- `is_asym_v<T>` = is it a _asym<T,N>?

Constructors:
- `()` = initialize elements to {}, aka 0 for numeric types.
- `(s)` = initialize with all elements set to `s`.
- `(x, y), (x, y, z), (x, y, z, w)` for dimensions 1-4, initialize with scalar values.
- `(function<Scalar(int i1, ...)>)` = initialize with a lambda that accepts the index of the matrix as a list of int arguments and returns the value for that index.
- `(function<Scalar(intN i)>)` = initialize with a lambda, same as above except the index is stored as an int-vector in `i`.
- `(tensor t)` = initialize from another tensor.  Truncate dimensions.  Uninitialized elements are set to {}.

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
- `elemMul(a,b), matrixCompMult(a,b), hadamard(a,b)` = per-element multiplication aka Hadamard product.
	$$elemMul(a,b)_I := a_I \cdot b_I$$
- `dot(a,b), inner(a,b)` = Frobenius dot.  Sum of all elements of a self-Hadamard-product.  Conjugation would matter if I had any complex support, but right now I don't.
	$$dot(a,b) := a^I \cdot b_I$$
- `lenSq(a)` = For vectors this is the length-squared.  It is a self-dot, for vectors this is equal to the length squared, for tensors this is the Frobenius norm (... squared? Math literature mixes up the definition of "norm" between the sum-of-squares and its square-root.).
	$$lenSq(a) := |a|^2 = a^I a_I$$
- `length(a)` = For vectors this is the length.  It is the sqrt of the self-dot.
	$$length(a) := |a| = \sqrt{a^I a_I}$$
- `normalize(a)` = For vectors this returns a unit.  It is a tensor divided by its sqrt-of-self-dot
	$$normalize(a) := a / |a|$$
- `distance(a,b)` = Length of the difference of two tensors.
	$$distance(a,b) := |b - a|$$
- `cross(a,b)` = 3D vector cross product.
	$${cross(a,b)_i} := {\epsilon_{ijk}} b^j c^k$$ 
- `outer(a,b), outerProduct(a,b)` = Tensor outer product.  Two vectors make a matrix.  A vector and a matrix make a rank-3.  Etc.  This also preserves storage optimizations, so an outer between a sym and a sym produces a sym-of-syms.
	$$outer(a,b)_{IJ} := a_I b_J$$
- `determinant(m)` = Matrix determinant, equal to `dot(cross(m.x, m.y), m.z)`.
	$$determinant(a) := det(a) = \epsilon_I {a^{i_1}}_1 {a^{i_2}}_2 {a^{i_3}}_3 ... {a^{i_n}}_n$$
- `inverse(m)` = Matrix inverse, for rank-2 tensors.
	$${inverse(a)^{i_1}}_{j_1} := \frac{1}{(n-1)! det(a)} \delta^I_J {a^{j_2}}_{i_2} {a^{j_3}}_{i_3} ... {a^{j_n}}_{i_n}$$
- `transpose<from=0,to=1>(a)` = Transpose indexes `from` and `to`.  This will preserve storage optimizations, so transposing 0,1 of a sym-of-vec will produce a sym-of-vec, but transposing 0,2 or 1,2 of a sym-of-vec will produce a vec-of-vec-of-vec.
	$$transpose(a)_{{i_1}...{i_p}...{i_q}...{i_n}} = a_{{i_1}...{i_q}...{i_p}...{i_n}}$$
- `trace(m)` = Matrix trace = Tensor contraction between two indexes.
	$$trace(a) = {a^i}_i$$
- `contract<i=0,j=1>(a), interior(a)` = Tensor contraction / interior product of indexes 'i' and 'j'. For rank-2 tensors where i=0 and j=1, `contract(t)` is equivalent to `trace(t)`.
- `diagonal(m)` = Matrix diagonal from vector.
	$${diagonal(a)_{ij} = \delta_{ij} \cdot a_i$$

## Familiar Types

Sorry GLSL, Cg wins this round:
- `floatN<N>` = N-dimensional float vector.
- `float2, float3, float4` = 1D, 2D, 3D float vector.
- `float2x2, float2x3, float2x4, float3x2, float3x3, float3x4, float4x2, float4x3, float4x4` = matrix types.
- `float2s2, float3s3, float4s4` = symmetric matrix types.
- `float2a2, float3a3, float4a4` = antisymmetric matrix types.
- ... same with bool, char, uchar, short, ushort, int, uint, float, double, ldouble, size, intptr, uintptr.
- `_vec2<T>, _vec3<T>, _vec4<T>` = templated fixed-size vectors.
- `_mat2x2<T>, _mat2x3<T>, _mat2x4<T>, _mat3x2<T>, _mat3x3<T>, _mat3x4<T>, _mat4x2<T>, _mat4x3<T>, _mat4x4<T>` = templated fixed-size matrices.
- `_sym2<T>, _sym3<T>, _sym4<T>` = templated fixed-size symmetric matrices.
- `_asym2<T>, _asym3<T>, _asym4<T>` = templated fixed-size antisymmetric matrices.

Depends on the "Common" project, for Exception, template metaprograms, etc.

TODO:
- finishing up antisym , it needs intN access, and a loooot of wrapper classes.  
	- once we have this, why not rank-3 rank-4 etc sym or antisym ... I'm sure there's some math on how to calculate the unique # of vars
- sym needs some index help when it's mixing accessors and normal derefs. one signature is Scalar&, the other is Accessor

- more flexible multiplication ... it's basically outer then contract of lhs last and rhs first indexes ... though I could optimize to a outer+contract-N indexes
	- for mul(A a, B b), this would be "expandStorage" on the last of A and first of B b, then "ReplaceScalar" the nesting-minus-one of A with the nesting-1 of B
- make a permuteOrder() function, have this "ExpandIthIndex<>" on all its passed indexes, then have it run across the permute indexes.
	- mind you that for transposes then you can respect symmetry and you don't need to expand those indexes.
	- make transpose a specialization of permuteIndexes()
- index notation summation?  mind you that it shoud preserve non-summed index memory-optimization structures.
- multiply as contraction of indexes
	- for multiply and for permute, some way to extract the flags for symmetric/not on dif indexes
	  so when i produce result types with some indexes moved/removed, i'll know when to expand symmetric into vec-of-vec
- shorthand those other longwinded GLSL names like "inverse"=>"inv", "determinant"=>"det", "transpose"=>"tr" "normalize"=>"unit"
- get contract() working (and proly matrix-mul too)
- TODO RemoveIndex<> doesn't sort yet, so the order you provide matters, but I just put sequence-sort in Common/Meta.h so yeah soon.

- more flexible exterior product (cross, wedge, determinant).  Generalize to something with Levi-Civita permutation tensor.
- asym is 2-rank totally-antisymmetric .. I should do a N-rank totally-antisymmetric and N-rank totally-symmetric 
	- then use it for an implementation of LeviCivita as constexpr
	- then use that for cross, determinant, inverse, wedge

- better function matching for derivatives?
- move secondderivative from Relativity to Tensor
- move covariantderivative from Relativity to Tensor
- move Hydro/Inverse.h's GaussJordan solver into Tensor/Inverse
- get rid fo the Grid class.  
	The difference between Grid and Tensor is allocation: Grid uses dynamic allocation, Tensor uses static allocation.
	Intead, make the allocator of each dimension a templated parameter: dynamic vs static.
	This will give dynamically-sized tensors all the operations of the Tensor template class without having to re-implement them all for Grid.
	This will allow for flexible allocations: a degree-2 tensor can have one dimension statically allocated and one dimension dynmamically allocated

- double check that I can't use "requires' with fields and put all vector-specialization in one class
- should I even keep separate member functions for .dot() etc?
- __complex__ support.  Especially in norms.
- preserve storage optimizations between tensor op tensor per-elem operations.  right now its just expanding all storage optimizations.

- change all those functions types to be is_tensor_v when possible.
- try to work around github's mathjax rendering errors.  looks fine on my own mathjax and on stackedit, but not on github.
