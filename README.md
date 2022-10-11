## Tensor Library

After using fixed-dimension vectors and tensors for a few decades, and having a few goes at designing a C++ math library around them,
and then getting into differential geometry and relativity, and trying to design a C++ math library around that,
this is the latest result.

I know I've put the word "Tensor" in the title.
Ever since the deep learning revolution in AI that computer scientists have come to believe that a "tensor" is an arbitrary dimensioned array of numbers, preferrably larger dimensioned and smaller-indexed.
But this library is moreso centered around "tensor" in the original definition, as "geometric object that lives in the tangent space at some point on a manifold and is invariant to coordinate transform."
That means I am designing this library is centered around compile-time sized small arrays and larger ranks/degrees (whatever the term is for the number of indexes).

The old and pre-C++11 and ugly version had extra math indicators like Upper<> and Lower<> for tracking variance, but I've done away with that now.
This version got rid of that and has added a lot of C++20 tricks.
So I guess overall this library is midway between a mathematician's and a programmer's and a physicist's implementation.

- Familiar vector and math support, 2D 3D 4D.
- Arbitrary-dimension, arbitrary-rank.
- Compressed storage for identity, symmetric, and antisymmetric tensor indexes.
- Lots of compile time and template driven stuff.
- C++20
- Some sort of GLSL half-compatability, though creative freedom where I want it.

## Reference:

### Tensors:
`tensor` is not a typename, but is a term I will use interchangeably for the various tensor storage types.  These currently include: `_vec`, `_ident`, `_sym`, `_asym`, `_symR`, `_asymR`.

### Vectors:
`_vec<type, dim>` = vectors:
- `.s` = `std::array<type>` = element std::array access.  Tempted to change it back to `type[] s` for ease of ue as pointer access... but I do like the ease of iterator use with `std::array`... hmm...
- for 1D through 4D: `.x .y .z .w`, `.s0 .s1 .s2 .s2` storage.
- `.subset<size>(index), .subset<size,index>()` = return a vector reference to a subset of this vector.

### Matrices:
`_mat<type, dim1, dim2>` = `_vec<_vec<type,dim2>,dim1>` = matrices:
- Right now indexing is row-major, so matrices appear as they appear in C, and so that matrix indexing `A.i.j` matches math indexing $A\_{ij}$.
	This disagrees with GL compatability, so you'll have to upload your matrices to GL transposed.

### Identity Matrix:
`_ident<type, dim>` = identity matrix.
This will only take up a single value of storage.  Specifying the internal storage type is still required, which enables `_ident` to be used in conjunction with outer products to produce optimized-storage tensors,
i.e. $a\_{ij} \otimes \delta\_{kl}$ = `outer( _mat<float,3,3>(...), _ident<float,3>(1) )` will produce a rank-4 tensor $c\_{ijkl} = a\_{ij} \cdot \delta\_{kl}$ which will only require 9 floats of storage, not 81 floats, as a naive outer product would require.

### Symmetric Matrices:
`_sym<type, dim>` = symmetric matrices:
- `.x_x .x_y .x_z .y_y .y_z .z_z .x_w .y_w .z_w .w_w` storage, `.y_x .z_x, .z_y` union'd access.

### Antisymmetric Matrices:
`_asym<type, dim>` = antisymmetric matrices:
- `.x_x() .w_w()` access methods

### Totally-symmetric tensors:
`_symR<type, dim, rank>` = totally-symmetric tensor.
The size of a totally-symmetric tensor storage is
the number of unique permutations of a symmetric tensor of dimension `d` and rank `r`,
which is `(d + r - 1) choose r`.

### Totally-antisymmetric tensors:
`_asymR<type, dim, rank>` = totally-antisymmetric tensor.
The size of a totally-antisymmetric tensor storage is
the number of unique permutations of an antisymmetric tensor of dimension `d` and rank `r`,
which is `d choose r`.
This means the Levi-Civita permutation tensor takes up exactly 1 float.  
Feel free to initialize this as the value 1 for Cartesian geometry or the value of $\sqrt{det(g\_{uv})}$ for calculations in an arbitrary manifold.

### Accessors:
An accessor is an object used for intermediate access to a tensor when it is not fully indexed.  For example, if you construct an object `auto a = float3s3()` to be a rank-2 symmetric 3x3 matrix,
and you decided to read only a single index: `a[i]`, the result will be an Accessor object which merely wraps the owner.  From here the Accessor can be indexed a second time: `a[i][j]` to return a reference to the `_sym`'s variable.

These Accessor objects can be treated like tensors for almost all intents and purposes.
They can be passed into other tensor math operations, tensor constructors (and by virtue of this, casting).
I even have tensor member methods that call back into the `Tensor::` namespace as I do other tensors.

I still don't have `+= -= *= /=` math operators for Accessors.  This is because they are tenuous enough for the non-Accessor tensor classes themselves
(since the non-= operator will often produce a different return type than the lhs tensor, so `a op = b; auto c = a;` in many cases cannot reproduce identical behavior to `auto c = a op b`.

### Tensor creation:
- `_tensor<type, dim1, ..., dimN>` = construct a rank-N tensor, equivalent to nested `_vec< ... , dim>`.
- `_tensorr<type, dim, rank>` = construct a tensor of rank-`rank` with all dimensions `dim`.
- `_tensori<type, I1, I2, I3...>` = construct a tensor with specific indexes vector storage and specific sets of indexes symmetric or antisymmetric storage.
	`I1 I2` etc are one of the following:
- - `index_vec<dim>` for a single index of dimension `dim`,
- - `index_ident<dim>` for rank-two identity indexes of dimension `dim`,
- - `index_sym<dim>` for two symmetric indexes of dimension `dim`,
- - `index_asym<dim>` for two antisymmetric indexes of dimension `dim`,
- - `index_symR<dim, rank>` for `rank` symmetric indexes of dimension `dim`,
- - `index_asymR<dim, rank>` for `rank` antisymmetric indexes of dimension `dim`.
	Ex: `_tensor<float, index_vec<3>, index_sym<4>, index_asym<5>>` is the type of a tensor $a\_{ijklm}$ where index i is dimension-3, indexes j and k are dimension 4 and symmetric, and indexes l and m are dimension 5 and antisymmetric.

### Tensor operators
- `operator += -= /=` = In-place per-element operations.
- `operator == !=` = Tensor comparison.  So long as the rank and dimensions match then this should evaluate successfully.
- `operator + - /` = Scalar/tensor, tensor/scalar, per-element tensor/tensor operations.
	Notice that `+ - /` run the risk of modifying the internal storage.
	If you add subtract or divide `_ident, _asym, _asymR` by a scalar, the type becomes promoted to `_sym, _mat, _tensorr`.
	Division is only on this list courtesy of divide-by-zero, otherwise division would've been safe for maintaining storage.
- `operator *`
	- Scalar/tensor, tensor/scalar for per-element multiplication.
	- Tensor/tensor multiplication is an outer then a contraction of the adjacent indexes.  Therefore:
		- `_vec * _vec` multiplication is a dot product.
		- `_vec * _mat` as row-multplication.
		- `_mat * _vec` as column-multiplication
		- `_mat * _mat` as matrix-multiplication.

### Tensor properties:
- `::This` = The current type.  Maybe not useful outside the class definition, but pretty useful within it when defining inner classes.
- `::Scalar` = Get the scalar type used for this tensor.
		Mind you, wrt the tensor library, 'Scalar' refers to the nested-most Inner class that is not a tensor.
		You could always use a vector-of-functions and I won't try to stop you, so long as you implement the necessary functions to perform whatever operations you are doing with it.
- `::Inner` = The next most nested vector/matrix/symmetric.
- `::Template<T, N>` = The template of this class, useful for nesting operations.
- `::rank` = For determining the tensor rank.  Vectors have rank-1, Matrices (including symmetric) have rank-2.
- `::dim<i>` = Get the i'th dimension size , where i is from 0 to rank-1.
- `::dims` = Get a int-vector with all the dimensions.  For rank-1 this is just an int.  Maybe I'll change it to be intN for all ranks, not sure.
- `::isSquare` = true if all dimensions match, false otherwise.  Yes that means vectors are square.  Maybe I'll change the name.
- `::localDim` = Get the dimension size of the current class, equal to `dim<0>`.
- `::numNestings` = Get the number of template-argument-nested classes.  Equivalent to the number of Inner's in `tensor::Inner::...::Inner`.
- `::count<i>` = Get the storage size of the i'th nested class.  i is from 0 to numNestings-1.
- `::localCount` = Get the storage size of the current class, equal to `count<0>`.
- `::Nested<i>` = Get the i'th nested type from our tensor type, where i is from 0 to numNestings-1.
- `::numNestingsToIndex<i>` = Gets the number of nestings deep that the index 'i' is located, where i is from 0 to rank.
	`numNestingsToIndex<0>` will always return 0, `numNestingsToIndex<rank>` will always return `numNestings`.
- `::indexForNesting<i>` = Get the index of the i'th nesting, where i is from 0 to `numNestings`.  `indexForNesting<numNestings>` will produce `rank`.
- `::InnerForIndex<i>` = Get the type associated with the i'th index, where i is from 0 to rank.
	Equivalent to `::Nested<numNestingsToIndex<i>>`  `_vec`'s 0 will point to itself, `_sym`'s and `_asym`'s 0 and 1 will point to themselves, all others drill down.
- `::ReplaceInner<T>` = Replaces this tensor's Inner with a new type, T.
- `::ReplaceNested<i,T>` = Replace the i'th nesting of this tensor with the type 'T', where i is from 0 to numNestings-1.
- `::ReplaceScalar<T>` = Create a type of this nesting of tensor templates, except with the scalar-most type replaced by T.
	Should be equivalent to `This::ReplaceNested<This::numNestings, T>`.
- `::ReplaceLocalDim<n>` = Replaces this tensor's localDim with a new dimension, n.
- `::ReplaceDim<i,n>` = Replace the i'th index dimension with the dimension, n.  If the dimensions already match then nothing is done.  If not then the stroage at this index is expanded.
- `::ExpandIthIndex<i>` = Produce a type with only the storage at the i'th index replaced with expanded storage.
	Expanded storage being a `_vec`-of-`_vec`-of...-`_vec`'s with nesting equal to the desired tensor rank.
	So a `_sym`'s ExpandIthIndex<0> or <1> would produce a `_vec`-of-`_vec`, a `_sym`-of-`_vec`'s ExpandIthIndex<2> would return the same type, and a `_vec`-of-`_sym`'s ExpandIthIndex<0> would return the same type.
- `::ExpandIndex<i1, i2, ...>` = Expand the all indexes listed.
- `::ExpandIndexSeq<std::integer_sequence<int, i1...>>` = Expands the indexes in the `integer_sequence<int, ...>`.
- `::ExpandAllIndexes<>` = Produce a type with all storage replaced with expanded storage.
	Expanded storage being a `_vec`-of-`_vec`-of...-`_vec`'s with nesting equal to the desired tensor rank.
	Equivalent to `T::ExpandIthIndex<0>::...::ExpandIthIndex<T::rank-1>`.  Also equivalent to an equivalent tensor with no storage optimizations, i.e. `_tensori<Scalar, dim1, ..., dimN>`.
- `::RemoveIthIndex<i>` = Removes the i'th index.  First expands the storage at that index, so that any rank-2's will turn into `_vec`s.  Then removes it.
- `::RemoveIndex<i1, i2, ...>` = Removes all the indexes, `i1` ..., from the tensor.

### Template Helpers (subject to change)
- `is_tensor_v<T>` = is it a tensor storage type?
- `is_vec_v<T>` = is it a `_vec<T,N>`?
- `is_ident_v<T>` = is it a `_ident<T,N>`?
- `is_sym_v<T>` = is it a `_sym<T,N>`?
- `is_asym_v<T>` = is it a `_asym<T,N>`?
- `is_symR_v<T>` = is it a `_symR<T,N>`?
- `is_asymR_v<T>` = is it a `_asymR<T,N>`?

### Constructors:
- `()` = initialize elements to {}, aka 0 for numeric types.
- `(s)` = initialize with all elements set to `s`.
- `(x, y), (x, y, z), (x, y, z, w)` for dimensions 1-4, initialize with scalar values.
- `(function<Scalar(int i1, ...)>)` = initialize with a lambda that accepts the index of the matrix as a list of int arguments and returns the value for that index.
- `(function<Scalar(intN i)>)` = initialize with a lambda, same as above except the index is stored as an int-vector in `i`.
- `(tensor t)` = initialize from another tensor.  Truncate dimensions.  Uninitialized elements are set to {}.

### Overloaded Indexing
- `(int i1, ...)` = dereference based on a list of ints.  In the case of `_tensori` or `_tensorr`, i.e. `_vec<_vec<_vec<...>>>` storage, this means $a\_{ij}$ in math = `a.s[i].s[j]` in code.
- `(intN i)` = dereference based on a vector-of-ints. Same convention as above.
- `[i1][i2][...]` = dereference based on a sequence of bracket indexes.  Same convention as above.
	Mind you that in the case of optimized storage being used this means the `[][]` indexing __DOES NOT MATCH__ the `.s[].s[]` indexing.
	In the case of optimized storage, for intermediate-indexes, an Accessor wrapper object will be returned.

### Iterating
- `.begin() / .end() / .cbegin() / .cend()` = read-iterators, for iterating over indexes (including duplicate elements in symmetric matrices).
- `.write().begin(), ...` = write-iterators, for iterating over the stored elements (excluding duplicates for symmetric indexes).
- read-iterators have `.index` as the int-vector of the index into the tensor.
- write-iterators have `.readIndex` as the int-vector lookup index and `.writeIndex` as the nested-storage int-vector.

### Swizzling
Swizzling is only available for expanded storage, i.e. `_vec` and compositions of `_vec` (like `_mat`), i.e. it is not available for `_sym` and `_asym`.
Swizzling will return a vector-of-references:
- 2D: `.xx() .xy() ... .wz() .ww()`
- 3D: `.xxx() ... .www()`
- 4D: `.xxxx() ... .wwww()`

### Functions
Functions are provided as `Tensor::` namespace or as member-functions where `this` is automatically padded into the first argument. 
- `dot(a,b), inner(a,b)` = Frobenius inner.  Sum of all elements of a self-Hadamard-product.  Conjugation would matter if I had any complex support, but right now I don't.
	- rank-N x rank-N -> rank-0.
	$$dot(a,b) := a^I \cdot b\_I$$
- `lenSq(a)` = For vectors this is the length-squared.  It is a self-dot, for vectors this is equal to the length squared, for tensors this is the Frobenius norm (... squared? Math literature mixes up the definition of "norm" between the sum-of-squares and its square-root.).
	- rank-N -> rank-0
	$$lenSq(a) := |a|^2 = a^I a\_I$$
- `length(a)` = For vectors this is the length.  It is the sqrt of the self-dot.
	- rank-N -> rank-0
	$$length(a) := |a| = \sqrt{a^I a\_I}$$
- `distance(a,b)` = Length of the difference of two tensors.
	- rank-N x rank-N -> rank-0:
	$$distance(a,b) := |b - a|$$
- `normalize(a)` = For vectors this returns a unit.  It is a tensor divided by its sqrt-of-self-dot
	- rank-N -> rank-N:
	$$normalize(a) := a / |a|$$
- `elemMul(a,b), matrixCompMult(a,b), hadamard(a,b)` = per-element multiplication aka Hadamard product.
	- rank-N x rank-N -> rank-N:
	$$elemMul(a,b)\_I := a\_I \cdot b\_I$$
- `cross(a,b)` = 3D vector cross product.
	- rank-1 dim-3 x rank-1 dim-3 -> rank-1 dim-3:
	$$cross(a,b)\_i := \epsilon\_{i j k} b^j c^k$$
- `outer(a,b), outerProduct(a,b)` = Tensor outer product.  The outer of a `_vec` and a `_vec` make a `_mat` (i.e. a `_vec`-of-`_vec`).
	The outer of a vector and a matrix make a rank-3.  Etc.  This also preserves storage optimizations, so an outer between a `_sym` and a `_sym` produces a `_sym`-of-`_sym`s.
	- rank-M x rank-N -> rank-(M+N):
	$$outer(a,b)\_{IJ} := a\_I b\_J$$
- `transpose<from=0,to=1>(a)` = Transpose indexes `from` and `to`.
	This will preserve storage optimizations, so transposing 0,1 of a `_sym`-of-`_vec` will produce a `_sym`-of-`_vec`,
	but transposing 0,2 or 1,2 of a `_sym`-of-`_vec` will produce a `_vec`-of-`_vec`-of-`_vec`.
	- rank-M -> rank-M, M >= 2
	$$transpose(a)\_{{i\_1}...{i\_p}...{i\_q}...{i\_n}} = a\_{{i\_1}...{i\_q}...{i\_p}...{i\_n}}$$
- `contract<m=0,n=1>(a), trace(a)` = Tensor contraction / interior product of indexes 'm' and 'n'. For rank-2 tensors where m=0 and n=1, `contract(t)` is equivalent to `trace(t)`.
	- rank-M -> rank-M-2 (for different indexes.  rank-M-1 for same indexes.). M >= 1
	$$contract(a) = \delta^{i\_m i\_n} a\_I$$
- `contractN<i=0,n=1>(a)` = Tensor contraction of indexes i ... i+n-1 with indexes i+n ... i+2n-1.
- `interior<n=1>(a)` = Interior product of neighboring n indexes.  I know a proper interior product would default n to `A::rank`.  Maybe later.  For n=1 this defaults to matrix-multiply.
- `makeSym(a)` = Create a symmetric version of tensor a.  Only works if t is square, i.e. all dimensions are matching.
	$$makeSym(a)\_I = a\_{(i\_1 ... i\_ n)} $$
- `makeAsym(a)` = Create an antisymmetric version of tensor a.  Only works if t is square, i.e. all dimensions are matching.
	$$makeAsym(a)\_I = a\_{[i\_1 ... i\_ n]} $$
- `wedge(a,b)` = The wedge product of tensors 'a' and 'b'.
	$$wedge(a,b)\_I = (a \wedge b)\_I = Alt (a \otimes b)\_I = a\_{[i\_1 ... i\_p} b\_{i\_{p+1} ... i\_{p+q}]}$$
- `hodgeDual(a)` = The Hodge-Dual of rank-k tensor 'a'.  This only operates on 'square' tensors.  If you really want to produce the dual of a scalar, just use `_asymR<>(s)`.
	$$hodgeDual(a)\_I = (\star a)_I = \frac{1}{k!} T\_J {\epsilon^J}\_I$$
- `diagonal<m=0>(a)` = Matrix diagonal from vector.  For tensors it takes one index and turns it into two.
	- rank-N -> rank-(N+1), N>=1:
	$$diagonal(a)\_I = \delta\_{i\_m i\_{m+1}} a\_{i\_1 ... i\_m i\_{m+2} ... i\_n}$$
- `determinant(m)` = Matrix determinant, equal to `dot(cross(m.x, m.y), m.z)`.
	- rank-2 -> rank-0:
	$$determinant(a) := det(a) = \epsilon\_I {a^{i\_1}}\_1 {a^{i\_2}}\_2 {a^{i\_3}}\_3 ... {a^{i\_n}}\_n$$
- `inverse(m)` = Matrix inverse, for rank-2 tensors.
	- rank-2 -> rank-2:
	$${inverse(a)^{i\_1}}\_{j\_1} := \frac{1}{(n-1)! det(a)} \delta^I\_J {a^{j\_2}}\_{i\_2} {a^{j\_3}}\_{i\_3} ... {a^{j\_n}}\_{i\_n}$$

### Familiar Types

Sorry GLSL, Cg wins this round:
- `floatN<N>` = N-dimensional float vector.
- `float2, float3, float4` = 1D, 2D, 3D float vector.
- `float2x2, float2x3, float2x4, float3x2, float3x3, float3x4, float4x2, float4x3, float4x4` = matrix of floats.
- `float2i2, float3i3, float4i4` = identity matrix of floats.
- `float2s2, float3s3, float4s4` = symmetric matrix of floats.
- `float2s2s2 float3s3s3 float4s4s4 float2s2s2s2 float3s3s3s3 float4s4s4s4` = totally-symmetric tensor of floats.
- `float2a2, float3a3, float4a4` = antisymmetric matrix of floats.
- `float3a3a3 float4a4a4 float4a4a4a4` = totally-antisymmetric tensor of floats.
- `floatNxN<dim>` = matrix of floats of size `dim`.
- `floatNiN<dim>` = identity matrix of float of size `dim`.
- `floatNsN<dim>` = symetric matrix of floats of size `dim`.
- `floatNaN<dim>` = antisymmetric matrix of floats of size `dim`.
- `floatNsR<dim, rank>` = totally-symmetric tensor of arbitrary dimension and rank.
- `floatNaR<dim, rank>` = totally-antisymmetric tensor of arbitrary dimension and rank.
- ... same with bool, char, uchar, short, ushort, int, uint, float, double, ldouble, size, intptr, uintptr.
- `_vec2<T>, _vec3<T>, _vec4<T>` = templated fixed-size vectors.
- `_mat2x2<T>, _mat2x3<T>, _mat2x4<T>, _mat3x2<T>, _mat3x3<T>, _mat3x4<T>, _mat4x2<T>, _mat4x3<T>, _mat4x4<T>` = templated fixed-size matrices.
- `_sym2<T>, _sym3<T>, _sym4<T>` = templated fixed-size symmetric matrices.
- `_asym2<T>, _asym3<T>, _asym4<T>` = templated fixed-size antisymmetric matrices.

## Quaternions:
`_quat` is the odd one out, where it does have a few of the tensor operations, but it is stuck at 4D.  Maybe I will implement Cayley-Dickson constructs later for higher dimension.
`_quat<type>` = Quaternion.  Subclass of `_vec4<type>`.
`operator *` = Quaternion multiplication.
- `quati, quatf, quatd` = integer, float, and double precision quaternions.

## Dependencies:
This project depends on the "Common" project, for Exception, template metaprograms, etc.

TODO:
- make a permuteIndexes() function, have this "ExpandIndex<>" on all its passed indexes, then have it run across the permute indexes.
	- mind you that for transposes then you can respect symmetry and you don't need to expand those indexes.
	- make transpose a specialization of permuteIndexes()
- index notation summation?  mind you that it shoud preserve non-summed index memory-optimization structures.
- shorthand those longwinded names like "inverse"=>"inv", "determinant"=>"det", "trace"=>"tr", "transpose"=>...? T? tr?  what? "normalize"=>"unit"

- more flexible exterior product (cross, wedge, determinant).  Generalize to something with Levi-Civita permutation tensor.
	- use `_asymR` for an implementation of LeviCivita as constexpr
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

- should I even keep separate member functions for .dot() etc?
- `__complex__` support.  Especially in norms.

- InnerForIndex doesn't really get the inner, it gets the index, where index = corresponds with This, so call it something like "TypeForIndex"
- call 'Nested' something else like 'Next', and 'numNestings' to 'numNext' ... since 'nested-class' is a term that could be mistaken with member-classes.
- or call 'Inner" "Next" or something?

- ReplaceDim and ReplaceLocalDim that take a int pack and insert that many new dimensions into that index' location.
- would be nice to insert the template to wedge into it, like `tuple<index_int<3>, index_sym<3>>`.
	like `InsertIndexes<index, index_vec<3>, index_sym<3> >` etc,
		which would insert the listed structure into the nest of tensors.  make sure 'index' was expanded, expand it otherwise.

- makeAsym when applied to syms return a zero-tensor.
	Same with makeSym applied to asyms.
	A zero tensor is just the `_ident` with its scalar initialized to zero.  so to make `_ident` true to name, maybe make `_ident` default initialize its value to 1?

- shorter names for `index_*` for easier building tensors.

- more tensor types:  maybe diagonalized rank-2 with N-DOF, where each diagonal element has a unique value.

- any way to optimize symmetries between non-neighboring dimensions?
	- like $t\_{ijkl} = a\_{ij} b\_{kl}$ s.t. i and k are symmetric and j and l are symmetric, but they are not neighboring.

- range-iterator as a function of the tensor, like `for (auto i : a.range) {`
	- then maybe see how it can be merged with read and write iterator? maybe?  btw write iterator is just a range iterator over the sequence `count<0>...count<numNestings-1>` with a different lookup
	- so make a generic multi-dim iterator that acts on `index_sequence`. then use it with read and write iters.
 
- TODO in C++23 operator[] can be variadic.
	so once C++23 comes around, I'm getting rid of all Accessors and only allowing exact references into tensors using [] or ().
	in fact, why don't I just do that now?
	that would spare me the need for `_sym`'s operator[int]
	but don't forget `_asym` still needs to return an AntiSymRef , whether we allow off-storage indexing or not.
	so meh i might as well keep it around?

- somehow for differing-typed tensors get operator== constexpr
- might do some constexpr loop unrolling stuff maybe.
- test case write tests should be writing different values and verifying

- make `_symR` primary index be `operator(int...)` and secondary `operator(_vec<int,N>)`.  This requires splitting off param-packs at a specific index.

- RemoveIthIndex of `_symR` or `_asymR` should preserve the (anti)symmetry of the remaining indexes.  Atm it just turns the whole thing into a expanded-tensor.  Tho I did handle this when ExpandIthIndex is called on the first or last of an (a)sym... 

- Note to self (and anyone else listening), while GitHub MarkDown handles `_`'s correctly within `` ` ``'s , it fails within MathJax `$`'s and `$$`'s which means you have to escape all your `_`'s within your MathJax as `\_`.

- eventually merge `_sym` and `_asym` with `_symR` and `_asymR` ... but don't til enough sorts/loops are compile-time.
		
- `operator+=` etc for AntiSymRef
