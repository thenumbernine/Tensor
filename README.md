## Differential Geometry Tensor Library

[![Donate via Stripe](https://img.shields.io/badge/Donate-Stripe-green.svg)](https://buy.stripe.com/00gbJZ0OdcNs9zi288)<br>
[![Donate via Bitcoin](https://img.shields.io/badge/Donate-Bitcoin-green.svg)](bitcoin:37fsp7qQKU8XoHZGRQvVzQVP8FrEJ73cSJ)<br>

After using fixed-dimension vectors and tensors for a few decades, and having a few goes at designing a C++ math library around them,
and then getting into differential geometry and relativity, and trying to design a C++ math library around that,
this is the latest result.

I know I've put the word "Tensor" in the title.
Ever since the deep learning revolution in AI computer scientists have come to believe that a "tensor" is an arbitrary dimensioned array of numbers, preferrably larger dimensioned and smaller-indexed.
This library is moreso centered around "tensor" in the original differential geometry definition: a geometric object that lives in the tangent space at some point on a manifold and is invariant to coordinate transforms.
This means I am designing this library centered around compile-time sized small arrays and larger ranks/degrees/grades (whatever the term is for the number of indexes).

The old and pre-C++11 and ugly version had extra math indicators like Upper<> and Lower<> for tracking valence, but I've done away with that now.
There was no programmatically functional reason to track it (unless I wanted to verify Einstein-index summation correctness, which I never got to), so I've just done away with it.
What that means is you'll have to keep track of upper/lower/tensor basis valence all by yourself, and do your own metric multiplying all by yourself.
Luckily the default tensor `*` operator is a outer+contraction, aka matrix-multiply in the rank-2 case, which means any contraction of tensor `a`'s last index with tensor `b`s first index under metric `g` can just be written as `a * g * b`.

This version has added a lot of C++20 tricks.
So I guess overall this library is midway between a mathematician's and a programmer's and a physicist's implementation.

- All headers and templates.  No source files to be compiled.  Don't get anxious over seeing those custom build scripts, this library is strictly headers, you can just copy and paste the files wherever you want.
- Familiar vector and types and functions for 2D 3D 4D, with support for arbitrary-dimension, arbitrary-rank.
- Lots of compile time and template driven stuff.
- C++20
- Some sort of GLSL half-compatability, though creative freedom where I want it.
- Compressed storage for identity, symmetric, and antisymmetric tensor indexes.  For example, a 3x3 symmetric matrix takes 6 floats.  A 3x3 antisymmetric matrix takes 3 floats.  The 2x2, 3x3x3, 4x4x4x4 etc rank-N dim-N Levi-Civita permutation tensors take up 1 float.


## Examples

Example of Einstein index summation notation / Ricci calculus. $a\_{ij} := \frac{1}{2} (a\_{ij} + a\_{ji}), c\_i := {b\_{ij}}^j, d\_i := a\_{ij} {b^{jk}}\_k$
```c++
Index<'i'> i;
Index<'j'> j;
Index<'k'> k;
auto a = float3x3({{1,2,3},{4,5,6},{7,8,9}});
// lazy-evaluation of swizzles, addition, subtraction
a(i,j) = .5 * (a(i,j) + a(j,i));
auto b = float3s3s3([](int i, int j, int k) -> float { return i+j+k; });
// mid-evaluation caching of traces and tensor-multiplies
auto c = b(i,j,j).assignI();
auto d = (a(i,j) * b(j,k,k)).assignI();
```

Example of using a totally-antisymmetric tensor for implementing the cross-product. $(a \times b)\_i := \epsilon\_{ijk} a^j b^k$ 
```c++
float3 cross(float3 a, float3 b) {
	// Create the Levi-Civita totally-antisymmetric permutation tensor for dim 3 rank 3 ... using up a single float of memory:
	constexpr auto LC = float3a3a3(1);
	static_assert(sizeof(LC) == sizeof(float));
	// cross(a,b)_i = ε_ijk * a^j * b^k
	return (LC * b) * a;
}
```

Same thing but using exterior algebra.  $a \times b = \star (a \wedge b)$
```c++
float3 cross(float3 a, float3 b) {
	// Create a 3x3 antisymmetric
	auto c = a.wedge(b);
	// ... storing only 3 floats. mind you c[i][j] , 0 <= i < 3, 0 <= j < 3 are all accessible
	static_assert(sizeof(a) == 3*sizeof(float));
	// Return the dual, mapping those 3 antisymmetric rank-2 floats onto 3 rank-1 floats:
	return hodgeDual(c);
}
```

Example of using a totally-antisymmetric tensor to construct a basis perpendicular to a vector:
```c++
float2x3 basis(float3 n) {
	// Return a 3x3 antisymmetric, only storing 3 values, equivalent of initializing a matrix with the cross products of 'n' and each basis vector
	auto d = hodgeDual(n);
	static_assert(sizeof(d) == 3*sizeof(float));
	// Done.
	// We now have 3 vectors perpendicular to 'n' stored the 3 rows/cols of 'd'.
	// But maybe 'n' is closer to one of them, in which case the cross tends to zero, so best to pick the largest two:
	auto sq = float3([&d](int i) -> float { return d(i).lenSq(); });
	auto n2 = (
			(sq.x > sq.y)
			? ((sq.x > sq.z) ? d.x : d.z)
			: ((sq.y > sq.z) ? d.y : d.z)
		).normalize();
	return float2x3({n2, n.cross(n2)});
}
```

Example of using the Hodge dual to compute the Frobenius inner product of a and b.  In the case of rank &gt; 1 the value is the inner product of antisymmetrized a and b. $\langle a, b \rangle = \star (a \wedge \star b)$
```c++
auto inner(auto const & a, auto const & b)
//here 'isSquare' means all dimension sizes match
requires (a.dims() == b.dims() && a.isSquare && b.isSquare)
{
	//totally-antisymmetric tensors are space-optimized,
	//so the storage of bstar will always be <= the storage of b
	auto bstar = b.dual();
	//at this point c should be a n-form for dimension n,
	//which will take up a single numeric value (while representing n distinct tensor indexes)
	auto c = a.wedge(bstar);
	//return c's dual, which is a 0-form scalar, times the input rank factorial.
	return c.dual() * factorial(a.rank);
}
```

Example: Implementing the Levi-Civita totally-antisymmetric tensor in an orthonormal metric.  $\epsilon\_{i\_1 ... i\_n} = 1$
```c++
// the 'NaR' suffix stands for N-dimensional, totally-antisymmetric, R-rank, and expects <N,R> to come in the template arguments.
auto LC = floatNaR<dim,dim>(1);
// uses 1 whole float of storage.
```

Example: Implementing the covariant valence Levi-Civita totally-antisymmetric tensor for an arbitrary metric $g\_{ij}$: $\epsilon\_{i\_1 ... i\_n} = \sqrt{|det(g\_{ij})|}, \epsilon^{i\_1 ... i\_n} = \frac{1}{\sqrt{|det(g\_{ij})|}}$
```c++
auto g = floatNsN<dim>( /* provide your metric here */ );
auto detg = g.determinant();
auto LC_lower = floatNaR<dim, dim>(sqrt(det));	// uses 1 whole float of storage.
auto LC_upper = LC_lower / detg;
```
Mind you `LC_lower[0][0]...[0]` and `LC_lower[dim-1][dim-1]...[dim-1]` and every possible index access between are all valid C++ expressions.  But yeah, just 1 float of storage.

Example: ... and using it to compute the generalized Kronecker delta tensor.  $\delta^{i\_1 ... i\_n}\_{j\_1 ... j\_n} = \epsilon\_{j\_1 ... j\_n} \epsilon^{i\_1 ... i\_n}$
```c++
auto KD = LC_lower.outer(LC_upper);
```
KD is now a rank-`2*dim` tensor of dimension `dim`.
Once again it is represented by just a single float.
Take note of the order of your outer product and therefore the order of your result's indexes.  In this case the generalized-Kronecker-delta lower indexes are first.

## API Reference:

### Tensors:
`tensor` is not a typename, but is a term I will use interchangeably for the various tensor storage types.  These currently include: `_vec`, `_zero`, `_ident`, `_sym`, `_asym`, `_symR`, `_asymR`.

### Vectors:
`_vec<type, dim>` = vectors:
- rank-1
- `.s` = `std::array<type>` = element `std::array` access.  Tempted to change it back to `type[] s` for ease of use as pointer access... but I do like the ease of iterator use with `std::array`... hmm...
- for 1D through 4D: `.x .y .z .w`, `.s0 .s1 .s2 .s2` storage.
- `.subset<size>(index), .subset<size,index>()` = return a vector reference to a subset of this vector.

### Matrices:
`_mat<type, dim1, dim2>` = `_vec<_vec<type,dim2>,dim1>` = matrices:
- rank-2
- Right now indexing is row-major, so matrices appear as they appear in C, and so that matrix indexing `A.i.j` matches math indexing $A\_{ij}$.
	This disagrees with GL compatability, so you'll have to upload your matrices to GL transposed.

Tensor/tensor operator result storage optimizations:
- `mat` + `T` = `mat`

### Zero Vector:
`_zero<type, dim>` = vectors of zeroes.  Requires only the inner type worth of storage.  Use nesting for arbitrary-rank, arbitrary-dimension zero tensors all for no extra storage.
- rank-1
Always returns zero.  This is a shorthand for simpliciations like symmetrizing antisymmetric indexes or antisymmetrizing symmetric indexes.
Currently the internal mechanics expect - and provide - tensors such that, if any internal storage is zero, then all will be zero.

Tensor/tensor operator result storage optimizations:
- `zero` + `T` = `T`

### Identity Matrix:
`_ident<type, dim>` = identity matrix.
- rank-2
This will only take up a single value of storage.  Specifying the internal storage type is still required, which enables `_ident` to be used in conjunction with outer products to produce optimized-storage tensors,
i.e. $a\_{ij} \otimes \delta\_{kl}$ = `outer( _mat<float,3,3>(...), _ident<float,3>(1) )` will produce a rank-4 tensor $c\_{ijkl} = a\_{ij} \cdot \delta\_{kl}$ which will only require 9 floats of storage, not 81 floats, as a naive outer product would require.
Notice that `_ident` is rank-2, i.e. represents 2 indexes.  Sorry, just regular Kronecker delta here, not generalized Kronecker delta.  Besides, that's antisymmetric anyways, so for its use check out `_asymR`.

Tensor/tensor operator result storage optimizations:
- `ident` + `zero` = `ident`
- `ident` + `ident` = `ident`
- `ident` + `sym` = `sym`
- `ident` + `asym` = `matrix`
- `ident` + `matrix` = `matrix`

### Symmetric Matrices:
`_sym<type, dim>` = symmetric matrices:
- rank-2
- `.x_x .x_y .x_z .y_y .y_z .z_z .x_w .y_w .z_w .w_w` storage, `.y_x .z_x, .z_y` union'd access.

Tensor/tensor operator result storage optimizations:
- `sym` + `zero` = `sym`
- `sym` + `ident` = `sym`
- `sym` + `sym` = `sym`
- `sym` + `asym` = `matrix`
- `sym` + `matrix` = `matrix`

### Antisymmetric Matrices:
`_asym<type, dim>` = antisymmetric matrices:
- rank-2
- `.x_x() ... .w_w()` access methods.
No access fields, sorry. I had the option of making half named access via fields (maybe the upper trianglular ones) and the other half methods, but decided for consistency's sake to just use methods everywhere.
Don't forget that an antisymmetric matrix, i.e. a k-form, can be represented as $a\_{i\_1 ... i\_k} e^{[i\_1} \otimes ... \otimes e^{i\_k]} = \frac{1}{k!} a\_{i\_1 ... i\_k} e^{i\_1} \wedge ... \wedge e^{i\_k}$ - mind your scale factors.

Tensor/tensor operator result storage optimizations:
- `asym` + `zero` = `asym`
- `asym` + `ident` = `matrix`
- `asym` + `sym` = `matrix`
- `asym` + `asym` = `asym`
- `asym` + `matrix` = `matrix`

### Totally-symmetric tensors:
`_symR<type, dim, rank>` = totally-symmetric tensor.
- rank-N
The size of a totally-symmetric tensor storage is
the number of unique permutations of a symmetric tensor of dimension `d` and rank `r`,
which is $ \begin{pmatrix} d + r - 1 \\ r \end{pmatrix} $

Tensor/tensor operator result storage works the same as `_sym`:

### Totally-antisymmetric tensors:
`_asymR<type, dim, rank>` = totally-antisymmetric tensor.
- rank-N
The size of a totally-antisymmetric tensor storage is
the number of unique permutations of an antisymmetric tensor of dimension `d` and rank `r`,
which is $\left( \begin{matrix} d \\ r \end{matrix} \right)$.
This means the Levi-Civita permutation tensor takes up exactly 1 float.
Feel free to initialize this as the value 1 for Cartesian geometry or the value of $\sqrt{det(g\_{uv})}$ for calculations in an arbitrary manifold.

Tensor/tensor operator result storage works the same as `_asym`:

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
- ... same with `bool, char, uchar, short, ushort, int, uint, float, double, ldouble, size, intptr, uintptr`.
- `_vec2<T>, _vec3<T>, _vec4<T>` = templated fixed-size vectors.
- `_mat2x2<T>, _mat2x3<T>, _mat2x4<T>, _mat3x2<T>, _mat3x3<T>, _mat3x4<T>, _mat4x2<T>, _mat4x3<T>, _mat4x4<T>` = templated fixed-size matrices.
- `_sym2<T>, _sym3<T>, _sym4<T>` = templated fixed-size symmetric matrices.
- `_asym2<T>, _asym3<T>, _asym4<T>` = templated fixed-size antisymmetric matrices.

### Tensor creation:
- `_tensor<type, dim1, ..., dimN>` = construct a rank-N tensor, equivalent to nested `_vec< ... , dim>`.
- `_tensorr<type, dim, rank>` = construct a tensor of rank-`rank` with all dimensions `dim`.
- `_tensorx<type, description...>` = construct a tensor using the following arguments to describe its indexes storage optimization:
- - any number = an index of this dimension.
- - `-'z', dim` = use a rank-1 zero-tensor of dimension `dim`.
- - `-'i', dim` = use a rank-2 identity-tensor of dimension `dim`.
- - `-'s', dim` = use a rank-2 symmetric-tensor of dimension `dim`.
- - `-'a', dim` = use a rank-2 antisymmetric-tensor of dimension `dim`.
- - `-'S', dim, rank` = use a rank-`rank` totally-symmetric-tensor of dimension `dim`.
- - `-'A', dim, rank` = use a rank-`rank` totally-antisymmetric-tensor of dimension `dim`.
	Ex: `_tensorx<float, -'i', 3, -'A', 4, 4>` produces $T\_{ijklm} = t \cdot \delta\_{ij} \cdot \epsilon\_{klm}$.
- `_tensori<type, I1, I2, I3...>` = construct a tensor with specific indexes vector storage and specific sets of indexes symmetric or antisymmetric storage.
	`I1 I2` etc are one of the following storage types:
- - `storage_vec<dim>` for a single index of dimension `dim`,
- - `storage_zero<dim>` for rank-one zero valued indexes of dimension `dim`.  (If any are `storage_zero` then all should be `storage_zero`, but I left it to be specified per-index so that it could have varying dimensions per-index.),
- - `storage_ident<dim>` for rank-two identity indexes of dimension `dim`,
- - `storage_sym<dim>` for two symmetric indexes of dimension `dim`,
- - `storage_asym<dim>` for two antisymmetric indexes of dimension `dim`,
- - `storage_symR<dim, rank>` for `rank` symmetric indexes of dimension `dim`,
- - `storage_asymR<dim, rank>` for `rank` antisymmetric indexes of dimension `dim`.
	Ex: `_tensori<float, storage_vec<3>, storage_sym<4>, storage_asym<5>>` is the type of a tensor $a\_{ijklm}$ where index i is dimension-3, indexes j and k are dimension 4 and symmetric, and indexes l and m are dimension 5 and antisymmetric.

- `tensorScalarTuple<Scalar, StorageTuple>` = same as `_tensori` except the storage arguments are passed in a tuple.
- `tensorScalarSeq<Scalar, integer_sequence>` = same as `_tensor` except the dimensions are passed as a sequence.

### Tensor Operators
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

### Constructors:
- `()` = initialize elements to {}, aka 0 for numeric types.
- `(s)` = initialize with all elements set to `s`.
- `(x, y), (x, y, z), (x, y, z, w)` for dimensions 1-4, initialize with scalar values.
- `{...}` = initialize with `initializer_list`.
- `(integer_sequence<T>)` = initialize with an integer-sequence.
- `(function<Scalar(int i1, ...)>)` = initialize with a lambda that accepts the index of the matrix as a list of int arguments and returns the value for that index.
- `(function<Scalar(intN i)>)` = initialize with a lambda, same as above except the index is stored as an int-vector in `i`.
- `(tensor t)` = initialize from another tensor.  Truncate dimensions.  Uninitialized elements are set to {}.

### Overloaded Subscript / Array Access
Array access, be it by `()`, `[]`, by integer sequence or by int-vectors, is equivalent to accessing the respective elements of the tensor.
- `(int i1, ...)` = dereference based on a list of ints.  In the case of `_tensori` or `_tensorr`, i.e. `_vec<_vec<_vec<...>>>` storage, this means $a\_{ij}$ in math = `a.s[i].s[j]` in code.
- `(intN i)` = dereference based on a vector-of-ints. Same convention as above.
- `[i1][i2][...]` = dereference based on a sequence of bracket indexes.  Same convention as above.
	
In the case of optimized storage it can potentially differ with accessing the raw data directly, and for this reason reading the `.s[i]` field will not always produce the same value as reading the `[i]` field.
In the case of `zero, ident, asym` there are fields that provide wrapper objects which are either represent zero or the negative of another value within the tensor storage.
```c++
auto a = float3a3();	//initialize 3x3 antisymmetric of floats
auto a01 = a(0,1);		//returns a float value
auto a10 = a(1,0);		//returns an AntiSymRef value which wraps a reference to stored a(0,1), and reads and writes its negative.
```

### Accessors:
In the case of optimized storage being used this means the `[][]` indexing does not match the `.s[].s[]` indexing.
For intermediate-indexes an Accessor wrapper object will be returned.
```c++
auto a = float3s3();	//initialize 3x3 symmetric of floats
auto b = a[i];			//gets a Rank2Accessor object which points back to the original a
auto c = b[j];			//reads a[i][j] and stores a float value in c
```

An accessor is an object used for intermediate access to a tensor when it is not fully indexed.  For example, if you construct an object `auto a = float3s3()` to be a rank-2 symmetric 3x3 matrix,
and you decided to read only a single index: `a[i]`, the result will be an Accessor object which merely wraps the owner.  From here the Accessor can be indexed a second time: `a[i][j]` to return a reference to the `_sym`'s variable.

These Accessor objects can be treated like tensors for almost all intents and purposes.
They can be passed into other tensor math operations, tensor constructors (and by virtue of this, casting).
I even have tensor member methods that call back into the `Tensor::` namespace as I do other tensors.

I still don't have `+= -= *= /=` math operators for Accessors.  This is because they are tenuous enough for the non-Accessor tensor classes themselves
(since the non-= operator will often produce a different return type than the lhs tensor, so `a op = b; auto c = a;` in many cases cannot reproduce identical behavior to `auto c = a op b`.

### Structure Binding:
...works:
```c++
auto x = float3(1,2,3);
auto [r,theta,phi] = x;
```

Of course with structure-binding comes implementations of `std::tuple_size`, `std::tuple_element`, and `Tensor::get` working with tensors.

### Iterating
- `.begin() / .end() / .cbegin() / .cend()` = read-iterators, for iterating over indexes (including duplicate elements in symmetric matrices).
- `.write().begin(), ...` = write-iterators, for iterating over the stored elements (excluding duplicates for symmetric indexes).
- read-iterators have `.index` as the int-vector of the index into the tensor.
- write-iterators have `.readIndex` as the int-vector lookup index and `.index` as the nested-storage int-vector.
Iterations acts over all scalars of the tensor:
```c++
// read-iterators:
auto t = _tensorx<float, 3, -'s', 4>; // construct a 3x4x4 tensor with 2nd two indexes symmetric.
// loop over all indexes for reading, including symmetric indexes (where duplicates are not stored separately)
for (auto t_ijk : t) {
	doSomething(t_ijk);	//do something with the t_ijk's element of t
}
// loop but you want to access the read-iterator's index
for (auto i = t_ijk.begin(); i != t_ijk.end(); ++i) {
	doSomethingElse(t_ijk, i.index);
}
// loop over indexes of uniquely stored objects:
for (auto & t_ijk : t.write()) {
	t_ijk = calcSomething();
}
// loop but you want to access the write-iterator's index
auto w = t_ijk.write();
for (auto i = w.begin(); i != w.end(); ++i) {
	t_ijk = calcSomethingElse(i.readIndex);
}
```

### Swizzling
Swizzling is only available for expanded storage, i.e. `_vec` and compositions of `_vec` (like `_mat`), i.e. it is not available for `_sym` and `_asym`.
Swizzling will return a vector-of-references:
- 2D: `.xx() .xy() ... .wz() .ww()`
- 3D: `.xxx() ... .www()`
- 4D: `.xxxx() ... .wwww()`

```c++
auto a = float3(1,2,3);
// The swizzle methods themselves returns a vector-of-references.
auto bref = x.yzx();
// Writing values?  TODO. Still in the works ...
// bref = float3(4,5,6); // In a perfect world this would assign swizzled values to x ... for now it just errors.  Something about the complications of defining objects' ctors and operator='s, and having the two not fight with one another.
// Reading values works fine:
float3 b = x.yzx();
```

### Index Notation
- `Index<char>` = create an index iterator object.  Yeah I did see FTensor/LTensor doing this and thought it was a good idea.
	I haven't read enough of the paper on FTensor / copied enough that I am sure my implementation's performance is suffering compared to it.

- Index permutations are lazy-evaluated.
- Tensor/Scalar and Scalar/Tensor operations are lazy-evaluated.
- Tensor/Tensor add sub and per-element divide is lazy-evaluated.
- Same references on the LHS and RHS is ok.
- Traces are fine.  If any trace is present in a tensor expression then it will be calculated immediately and cached rather than lazy-evaluated.
	Traces producing a scalar can be used immediately, i.e. `float3x3 a; a(i,i);` will produce a float.  Traces producing a tensor will still need to be `.assign()`ed.
- Tensor-tensor multiplication works, and also caches mid-expression-evaluation.
- LHS typed assignment:
```c++
float3x3 a = ...;
float3a3 b; b(i,j) = (a(i,j) - a(j,i)) / 2.f;
```
- RHS typed assignment into specified return type:
```c++
float3x3 a = ...;
auto b = ((a(i,j) - a(j,i)) / 2.f).assignR<float3a3>(i,j);
```
- RHS typed assignment into implied return type based on specified assignment indexes:
```c++
float3x3 a = ...;
auto b = ((a(i,j) - a(j,i)) / 2.f).assign(i,j);
```
- RHS typed assignment into implied return type with implied index order.
This order is the order of non-summed indexes, and in the case of binary operations it is the first term's indexes.
```c++
float3x3 a = ...;
auto b = ((a(i,j) - a(j,i)) / 2.f).assignI();
```
- RHS type assignment right now will use an expanded tensor, so storage optimizations get lost:

Maybe I will merge assign, assignR, assignI into a single ugly abomination which is just the call operator,
such that if you pass it a specific template arg (can you do that?) it uses it as a return type, otherwise it infers from the indexes you pass it, otherwise if no indexes then it just uses the current index form of the expression as-is.

### Mathematics Functions
Functions are described using [Ricci Calculus](https://en.wikipedia.org/wiki/Ricci_calculus), though no meaning is assigned to upper or lower valence of tensor objects.  As stated earlier, you are responsible for all metric applications.
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
- `contract<m=0,n=1>(a), trace(a)` = Tensor contraction / interior product of indexes 'm' and 'n'. For rank-2 tensors where m=0 and n=1, `contract(t)` is equivalent to a matrix trace.
	- rank-M -> rank-M-2 (for different indexes.  rank-M-1 for same indexes.). M >= 1
	$$contract(a) = \delta^{i\_m i\_n} a\_I$$
- `contractN<i=0,n=1>(a)` = Tensor contraction of indexes i ... i+n-1 with indexes i+n ... i+2n-1.
	$${contractN(a)^I}\_J = {a^{I K}}\_{K J}, |I| = i, |K| = n$$
- `interior<n=1>(a,b)` = Interior product of neighboring n indexes.  I know a proper interior product would default n to `A::rank`.  Maybe later.  For n=1 this behaves the same a matrix-multiply.
	$${interior(a,b)^I}\_J = a^{I K} b\_{K J}, |K| = n$$
- `makeSym(a)` = Create a symmetric version of tensor a.  Only works if t is square, i.e. all dimensions are matching.  If the input is a tensor with any antisymmetric indexes then the result will be a zero-tensor.
	$$makeSym(a)\_I = a\_{(i\_1 ... i\_ n)} $$
- `makeAsym(a)` = Create an antisymmetric version of tensor a.  Only works if t is square, i.e. all dimensions are matching.  If the input is a tensor with any symmetric indexes then the result will be a zero-tensor.
	$$makeAsym(a)\_I = a\_{[i\_1 ... i\_ n]} $$
- `wedge(a,b)` = The wedge product of tensors 'a' and 'b'.  Notice that this antisymmetrizes the input tensors.
	$$wedge(a,b)\_I = (a \wedge b)\_I = Alt (a \otimes b)\_I = a\_{[i\_1 ... i\_p} b\_{i\_{p+1} ... i\_{p+q}]}$$
- `hodgeDual(a), dual(a)` = The Hodge-Dual of rank-k tensor 'a'.
	This only operates on 'square' tensors.
	If you really want to produce the dual of a scalar, just use `_asymR<>(s)`.
	Also it does not apply metrics.
	It will assume your input tensor is in $\sharp$ valence, and produce a tensor in $\flat$ valence.
	It also will assume a unit weight of the Levi-Civita permutation tensor, so if you happen to prefer your L.C. tensors to have weight $\sqrt{|g|}$ then you will have to multiply by this yourself.
	Notice that this antisymmetrizes the input tensor.
	$$hodgeDual(a)\_I = (\star a)_I = \frac{1}{k!} a^J \epsilon\_{JI}$$
- `diagonal<m=0>(a)` = Matrix diagonal from vector.  For tensors it takes one index and turns it into two.
	- rank-N -> rank-(N+1), N>=1:
	$$diagonal(a)\_I = \delta\_{i\_m i\_{m+1}} a\_{i\_1 ... i\_m i\_{m+2} ... i\_n}$$
- `determinant(m)` = Matrix determinant.
		For 2D this is equivalent to `_asymR<T,2,2>(1) * m.x * m.y`.
		For 3D this is equal to `dot(cross(m.x, m.y), m.z)`, i.e. `_asymR<T,3,3>(1) * m.x * m.y * m.z`.
	- rank-2 -> rank-0:
	$$determinant(a) := det(a) = \epsilon\_I {a^{i\_1}}\_1 {a^{i\_2}}\_2 {a^{i\_3}}\_3 ... {a^{i\_n}}\_n$$
- `inverse(m)` = Matrix inverse, for rank-2 tensors.
	- rank-2 -> rank-2:
	$${inverse(a)^{i\_1}}\_{j\_1} := \frac{1}{(n-1)! det(a)} \delta^I\_J {a^{j\_2}}\_{i\_2} {a^{j\_3}}\_{i\_3} ... {a^{j\_n}}\_{i\_n}$$

### Familiar OpenGL Functions

Each of these will return a 4x4 matrix of its respective scalar type.

- `translate<T>(_vec<T,3> t)` = returns a translation matrix.
	Based on `glTranslate`.
- `scale<T>(_vec<T,3> s)` = returns a scale matrix.
	Based on `glScale`.
- `rotate<T>(T angle, _vec<T,3> axis)` = returns a rotation matrix.
	Based on `glRotate`.
	`angle` is in radians, another break from OpenGL.
- `lookAt<T>(_vec<T,3> eye, _vec<T,3> center, _vec<T,3> up)` = returns a view matrix at `eye` looking at `center` with the up direction `up`.
	Based on `gluLookAt`.
- `frustum<T>(T left, T right, T bottom, T top, T near, T far)` = returns a frustum perspective matrix.
	Based on `glFrustum`.
- `perspective<T>(T fovY, T aspectRatio, T near, T far)` = returns a frustum perspective with the specified field-of-view and aspect-ratio.
	Based on `gluPerspective`.
	`fovY` is in radians.
- `ortho<T>(T left, T right, T bottom, T top, T near, T far)` = returns an ortho perspective matrix.
	Based on `glOrtho`.
- `ortho2D<T>(T left, T right, T bottom, T top)` = returns an ortho perspective matrix with default unit Z range.
	Based on `gluOrtho2D`.

### Support Functions
- `expand()` = convert the tensor to its expanded storage.  The type will be the same as `::ExpandAllIndexes<>`.

### Tensor properties:
- `::This` = The current type.  Maybe not useful outside the class definition, but pretty useful within it when defining inner classes.
- `::Scalar` = Get the scalar type used for this tensor.
		Mind you, wrt the tensor library, 'Scalar' refers to the nested-most Inner class that is not a tensor.
		You could always use a vector-of-functions and I won't try to stop you, so long as you implement the necessary functions to perform whatever operations you are doing with it.
- `::Inner` = The next most nested vector/matrix/symmetric.
- `::Template<T, N[, R]>` = The template of this class, useful for nesting operations.
- `::rank` = For determining the tensor rank.  Vectors have rank-1, Matrices (including symmetric) have rank-2.
- `::dimeq` = Sequence mapping the i'th index to the dimension of the i'th index.
- `::dim<i>` = Get the i'th dimension size , where i is from 0 to rank-1.
- `::dims` = Get a int-vector with all the dimensions.
- `::isSquare` = true if all dimensions match, false otherwise.  Yes that means vectors are square.  Maybe I'll change the name.
- `::localDim` = Get the dimension size of the current class, equal to `dim<0>`.
- `::numNestings` = Get the number of template-argument-nested classes.  Equivalent to the number of Inner's in `tensor::Inner::...::Inner`.
- `::countseq` = A sequence of the storage size of each nesting.
- `::count<i>` = Get the storage size of the i'th nested class.  i is from 0 to numNestings-1.
- `::localCount` = Get the storage size of the current class, equal to `count<0>`.
- `::totalCount` = The product of all counts of all nestings.  This times `sizeof(Scalar)` will equal the sizeof `This` tensor.
- `::LocalStorage` = The associated storage type used for building this tensor's nesting with `_tensori`.
- `::StorageTuple` = A tuple of all nestings' `LocalStorage`, such that `_tuplei<Scalar, ` ... then all types of `StorageTuple` ... `>` will produce This.  Equivalently, `tensorScalarTuple<Scalar, StorageTuple>` will produce `This`.
- `::NestedPtrTuple` = Tuple of pointers to the `Nested<i>`'th classes, where the 0'th is `This`, i.e., the current class, and the `numNestings`'th is the `Scalar`.  I had to use pointers because this is a member type, so it cannot contain itself.
- `::NestedPtrTensorTuple` = Same as above but without the last `Scalar` element, such that all its types are tensor types.
- `::Nested<i>` = Get the i'th nested type from our tensor type, where i is from 0 to numNestings-1.
- `::numNestingsToIndex<i>` = Gets the number of nestings deep that the index 'i' is located, where i is from 0 to rank.
	`numNestingsToIndex<0>` will always return 0, `numNestingsToIndex<rank>` will always return `numNestings`.
- `::indexForNesting<i>` = Get the index of the i'th nesting, where i is from 0 to `numNestings`.  `indexForNesting<numNestings>` will produce `rank`.
- `::InnerPtrTuple` = Tuple mapping the i'th tuple element to the i'th index associated nested Inner type. Tuple size is `rank`+1 (the +1 is for Scalar at the end).
- `::InnerPtrTemplateTuple` = Same as above but without Scalar at the end.  Useful for tuple operations.
- `::InnerForIndex<i>` = Get the type associated with the i'th index, where i is from 0 to rank.  Equivalent to getting the i'th element from `InnerPtrTuple`.
	Equivalent to `::Nested<numNestingsToIndex<i>>`  `_vec`'s 0 will point to itself, `_sym`'s and `_asym`'s 0 and 1 will point to themselves, all others drill down.
- `::ReplaceInner<T>` = Replaces this tensor's Inner with a new type, T.
- `::ReplaceNested<i,T>` = Replace the i'th nesting of this tensor with the type 'T', where i is from 0 to numNestings-1.
- `::ReplaceScalar<T>` = Create a type of this nesting of tensor templates, except with the scalar-most type replaced by T.
	Should be equivalent to `This::ReplaceNested<This::numNestings, T>`.
- `::ReplaceLocalDim<n>` = Replaces this tensor's localDim with a new dimension, n.
- `::ReplaceDim<i,n>` = Replace the i'th index dimension with the dimension, n.  If the dimensions already match then nothing is done.  If not then the stroage at this index is expanded.
- `::ExpandLocalStorage<i>` = Produce a tuple of storage types representing our type with the i'th local index undone from storage optimizations, where i is from 0 to `localRank-1`.  Un-expanded indexes storage is still preserved.
- `::ExpandIthIndexStorage<i>` = Produce a tuple of storage types but with its i'th index expanded, where i is from 0 to `rank-1`.
- `::ExpandIthIndex<i>` = Produce a tensor type with the storage at the i'th index replaced with expanded storage.
	Expanded storage being a `_vec`-of-`_vec`-of...-`_vec`'s with nesting equal to the desired tensor rank.
	So a `_sym`'s ExpandIthIndex<0> or <1> would produce a `_vec`-of-`_vec`, a `_sym`-of-`_vec`'s ExpandIthIndex<2> would return the same type, and a `_vec`-of-`_sym`'s ExpandIthIndex<0> would return the same type.
- `::ExpandIndex<i1, i2, ...>` = Expand the all indexes listed.
- `::ExpandIndexSeq<std::integer_sequence<int, i1...>>` = Expands the indexes in the `integer_sequence<int, ...>`.
- `::ExpandAllIndexes<>` = Produce a type with all storage replaced with expanded storage.
	Expanded storage being a `_vec`-of-`_vec`-of...-`_vec`'s with nesting equal to the desired tensor rank.
	Equivalent to `T::ExpandIthIndex<0>::...::ExpandIthIndex<T::rank-1>`.  Also equivalent to an equivalent tensor with no storage optimizations, i.e. `_tensori<Scalar, dim1, ..., dimN>`.
- `::RemoveIthNestedStorage<i>` = Produce a tuple of storage types but with the i'th element removed.
- `::RemoveIthNesting<i>` = Produce a tensor but with the i'th nesting removed.
- `::RemoveIthIndexStorage<i>` = Produce a tuple of storage types but with the i'th index expanded and i'th index storage removed.
- `::RemoveIthIndex<i>` = Removes the i'th index.  First expands the storage at that index, so that any rank-2's will turn into `_vec`s.  Then removes it.
- `::RemoveIndex<i1, i2, ...>` = Removes all the indexes, `i1` ..., from the tensor.
- `ReplaceWithZero<T>` = Returns a type with matching rank and dimensions, but all nestings are zeroes.  The result is fully-expanded so the nesting count matches the rank.

### Template Helpers
- `is_tensor_v<T>` = is it any tensor storage type?
- `is_vec_v<T>` = is it a `_vec<T,N>`?
- `is_zero_v<T>` = is it a `_zero<T,N>`?
- `is_ident_v<T>` = is it a `_ident<T,N>`?
- `is_sym_v<T>` = is it a `_sym<T,N>`?
- `is_asym_v<T>` = is it a `_asym<T,N>`?
- `is_symR_v<T>` = is it a `_symR<T,N>`?
- `is_asymR_v<T>` = is it a `_asymR<T,N>`?
- `is_quat_v<T>` = is it a `_quat<T>`?

## Quaternions:
`_quat` is the odd class out, where it does have a few of the tensor operations, but it is stuck at 4D.  Maybe I will implement Cayley-Dickson constructs later for higher dimension.
- `_quat<type>` = Quaternion.  Subclass of `_vec4<type>`.  This means they have the familiar fields `.x .y .z` for imaginary components and `.w` for real component.
- `quati, quatf, quatd` = integer, float, and double precision quaternions.

Quaternion Members and Methods:
- `::vec3` = the equivalent vec3 type which this quaternion is capable of rotating.
- `_quat operator*(_quat, _quat)` = multiplication between quaternions is defined as quaternion multiplication.
- `_quat static _quat::mul(_quat, _quat)` = helper function for quaternion multiplication.
- `_quat operator*=(_quat)` = in-place quaternion multiplication.
- `_quat .conjugate()` = returns the conjugate.  For unit quaternions this is the same as the inverse.
- `_quat .inverse()` = returns the inverse, i.e. the conjugate divided by the length-squared.
- `_quat .fromAngleAxis()` = Same as an quaternion exponential.  Takes a quaternion with the real/w component equal to the number of radians to rotate and x y z set to the axis to rotate along. Returns a quaternion which would produce that rotation.
- `_quat .toAngleAxis()` = Same as a quaternion logarithm.  Converts a quaternion from rotation space to angle-axis space.
- `static Scalar _quat::angleAxisEpsilon` = set this epsilon for when the from-angle-axis function should consider a rotation too small (would-be axis of zero) and avoid performing a normalization on the axis that would otherwise produce NaNs.
- `vec3 .rotate(vec3)` = rotate the vector by `this` quaternion.
- `vec3 .xAxis()` = return the x-axis of `this` quaternion's orientation.
- `vec3 .yAxis()` = return the y-axis of `this` quaternion's orientation.
- `vec3 .zAxis()` = return the z-axis of `this` quaternion's orientation.
- `_quat normalize(_quat)` = returns a normalized version of this quaternion.

## Dependencies:
This project depends on my "[Common](https://github.com/thenumbernine/Common)" project, for Exception, template metaprograms, etc.

## Literature Reference:

Any code lifted from a stackexchange or anywhere else will have an accompanying comment of where it came from.

Equally poorly referenced are the differential geometry notes of mine that I am basing this library on, found here: [https://thenumbernine.github.io/](https://thenumbernine.github.io/)

Likewise these notes' math references can be found here: [https://thenumbernine.github.io/math/Differential%20Geometry/sources.html](https://thenumbernine.github.io/math/Differential%20Geometry/sources.html).

Also: Lundy, "Implementing a High Performance Tensor Library", [https://wlandry.net/Presentations/FTensor.pdf](https://wlandry.net/Presentations/FTensor.pdf).

## TODO:

- make a permuteIndexes() function, have this "ExpandIndex<>" on all its passed indexes, then have it run across the permute indexes.
	- mind you that for transposes then you can respect symmetry and you don't need to expand those indexes.
	- make transpose a specialization of permuteIndexes()
	- this is already done in index notation assignments.  TODO make them compile-time.
- Does the index-notation summation preserve non-summed index memory-optimization structures?  I don't think it does.
- shorthand those longwinded names like "inverse"=>"inv", "determinant"=>"det", "trace"=>"tr", "transpose"=>...? T? tr?  what? "normalize"=>"unit"

- Add LeviCivita to the API using `constexpr _asymR`.
- Add KroneckerDelta to the API using `constexpr _ident`.
- Add GeneralizedKroneckerDelta using `constexpr _asymR` $\otimes$ `_asymR`.

- then use these for cross, determinant, inverse, wedge
- once index notation is finished that might be most optimal for implementations.

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

- To make `_ident` true to name, maybe make `_ident` default initialize its value to 1? Not sure...

- more tensor types:  maybe diagonalized rank-2 with N-DOF, where each diagonal element has a unique value.

- any way to optimize symmetries between non-neighboring dimensions?
	- like $t\_{ijkl} = a\_{ij} b\_{kl}$ s.t. i and k are symmetric and j and l are symmetric, but they are not neighboring.
 
- somehow for differing-typed tensors get operator== constexpr
- might do some constexpr loop unrolling stuff maybe.
- test case write tests should be writing different values and verifying

- make `_symR::operator(int...)` use perfect-forwarding.
- make `_asymR::operator(int...)` be primary and `::operator(_vec<int,N>)` be secondary.

- eventually merge `_sym` and `_asym` with `_symR` and `_asymR` ... but don't til enough sorts/loops are compile-time.

- wedge(a,b) should accept non-tensors, or at least the scalar of whichever is the tensor
- outer(a,b) too?  both turn into scalar muls.
- dual(a) too?  but it would require a default rank==k to turn a scalar into k-form of.

- make innerForIndexSeq which is a sequence mapping index to nesting #

- `operator[]` that takes a single intN?  or is that just redundant at this point?

- More tensor operators?  For integral Scalar types?  `<< >> & | ^ ~ && || ! % ?:`.  For tensors and index-notation.

- assign dimension size and offset to Index

- C++23 operator[] can be variadic.
	so once C++23 comes around, I'm getting rid of all Accessors and only allowing exact references into tensors using [] or ().
	in fact, why don't I just do that now?
	that would spare me the need for `_sym`'s operator[int]
	but don't forget `_asym` still needs to return an AntiSymRef , whether we allow off-storage indexing or not.
	so meh i might as well keep it around?

- C++23 has derived-this return-types, so no more need to crtp everything (RangeIterator and its classes in RangeObj, ReadIterator, WriteIterator)

- Note to self (and anyone else listening), while GitHub MarkDown handles `_`'s correctly within `` ` ``'s , it fails within MathJax `$`'s and `$$`'s which means you have to escape all your `_`'s within your MathJax as `\_`.
