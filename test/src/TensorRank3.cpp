#include "Test/Test.h"

/*
static_asserts for templates

order of tests, for each tested type:
static_asserts
default ctor
parenthesis ctor
list ctor
verify fields .x ... .s0 ...
verify .s[] storage
verify index access...
	all permutations of (i)(j)(k)(l)... to [i][j][k][l]...
	... of neighboring ()'s, all possible mergings of indexes into (int...)'s
	... of merged (int...)'s, also (intN)
	... of []'s, also .s[] (for non-optimized storage only)
ctor(scalar)
ctor(lambda(int))
ctor(lambda(intN))
ctor based on casting from another tensor
read iterator (with const)
write iterator (with const)
TODO test cbegin / cend
TODO implement rbegin / rend / crbegin / crend
subtensor field access based on .s[] (for matrices and rank>1 that aren't storage-optimized)
subset<n,i>() and subset<n>(i) access
swizzle
operators: == != += -= *= /= + - * /
string/stream operators: to_string and operator<<

math functions:
elemMul, hadamard, matrixCompMult
dot, inner
lenSq
length
distance
normalize
cross
outer, outerProduct
transpose
contract, interior, trace
diagonal
determinant
inverse
*/




namespace StaticTest1 {
	using namespace Tensor;
	using namespace std;

	static_assert(is_same_v<_tensorr<int,3,1>, _tensor<int,3>>);
	static_assert(is_same_v<_tensorr<int,3,2>, _tensor<int,3,3>>);
	static_assert(is_same_v<_tensorr<int,3,3>, _tensor<int,3,3,3>>);
	static_assert(is_same_v<_tensorr<int,3,4>, _tensor<int,3,3,3,3>>);
	
	static_assert(_vec<int,3>::template numNestingsToIndex<0> == 0);
	static_assert(_tensor<int,3,3>::template numNestingsToIndex<0> == 0);
	static_assert(_tensor<int,3,3>::template numNestingsToIndex<1> == 1);
	static_assert(_tensor<int,3,3,3>::template numNestingsToIndex<0> == 0);
	static_assert(_tensor<int,3,3,3>::template numNestingsToIndex<1> == 1);
	static_assert(_tensor<int,3,3,3>::template numNestingsToIndex<2> == 2);
	static_assert(_sym<int,3>::template numNestingsToIndex<0> == 0);
	static_assert(_sym<int,3>::template numNestingsToIndex<1> == 0);
	static_assert(_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::template numNestingsToIndex<0> == 0);
	static_assert(_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::template numNestingsToIndex<1> == 1);
	static_assert(_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::template numNestingsToIndex<2> == 1);
	static_assert(_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::template numNestingsToIndex<3> == 2);

	static_assert(is_same_v<_vec<int,3>::ExpandIthIndex<0>, _tensor<int,3>>);
	static_assert(is_same_v<_tensor<int,3,3>::ExpandIthIndex<0>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_tensor<int,3,3>::ExpandIthIndex<1>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_sym<int,3>::ExpandIthIndex<0>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_sym<int,3>::ExpandIthIndex<1>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_asym<int,3>::ExpandIthIndex<0>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_asym<int,3>::ExpandIthIndex<1>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_tensor<int,3,3,3>::ExpandIthIndex<0>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensor<int,3,3,3>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensor<int,3,3,3>::ExpandIthIndex<2>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_sym<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_asym<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_sym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_sym<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_sym<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_sym<3>>::ExpandIthIndex<2>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_asym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_asym<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_asym<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_asym<3>>::ExpandIthIndex<2>, _tensorr<int,3,3>>);
	static_assert(is_same_v<_tensor<int,3,3,3,3>::ExpandIthIndex<0>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensor<int,3,3,3,3>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensor<int,3,3,3,3>::ExpandIthIndex<2>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensor<int,3,3,3,3>::ExpandIthIndex<3>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<3>, _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<3>, _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_sym<3>,index_vec<3>> >);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::ExpandIthIndex<2>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_sym<3>,index_vec<3>>::ExpandIthIndex<3>, _tensori<int,index_vec<3>,index_sym<3>,index_vec<3>> >);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_asym<3>,index_vec<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_asym<3>,index_vec<3>> >);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_asym<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_asym<3>,index_vec<3>>::ExpandIthIndex<2>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_asym<3>,index_vec<3>>::ExpandIthIndex<3>, _tensori<int,index_vec<3>,index_asym<3>,index_vec<3>> >);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>::ExpandIthIndex<1>, _tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>::ExpandIthIndex<2>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>::ExpandIthIndex<3>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>::ExpandIthIndex<1>, _tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>::ExpandIthIndex<2>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>::ExpandIthIndex<3>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_sym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_sym<3>>::ExpandIthIndex<1>, _tensori<int,index_vec<3>,index_vec<3>,index_sym<3>>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_sym<3>>::ExpandIthIndex<2>, _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_sym<3>,index_sym<3>>::ExpandIthIndex<3>, _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_asym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_asym<3>>::ExpandIthIndex<1>, _tensori<int,index_vec<3>,index_vec<3>,index_asym<3>>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_asym<3>>::ExpandIthIndex<2>, _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_asym<3>>::ExpandIthIndex<3>, _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>>);
	
	static_assert(is_same_v<_vec<int,3>::ExpandAllIndexes<>, _vec<int,3>>);
	static_assert(is_same_v<_tensor<int,3,3>::ExpandAllIndexes<>, _tensor<int,3,3>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_asym<3>>::ExpandAllIndexes<>, _tensorr<int,3,4>>);
	
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_asym<3>>::ExpandIndex<0,1,2,3>, _tensorr<int,3,4>>);
	static_assert(is_same_v<_tensori<int,index_asym<3>,index_asym<3>>::ExpandIndexSeq<integer_sequence<int,0,1,2,3>>, _tensorr<int,3,4>>);

	static_assert(is_same_v<int3::Nested<0>, int3>);
	static_assert(is_same_v<int3::Nested<1>, int>);
	static_assert(is_same_v<int3s3::Nested<0>, int3s3>);
	static_assert(is_same_v<int3s3::Nested<1>, int>);
	static_assert(is_same_v<int3a3::Nested<0>, int3a3>);
	static_assert(is_same_v<int3a3::Nested<1>, int>);
	static_assert(is_same_v<int3x3::Nested<0>, int3x3>);
	static_assert(is_same_v<int3x3::Nested<1>, int3>);
	static_assert(is_same_v<int3x3::Nested<2>, int>);

	//only do the RemoveIthNesting on non-opt storage (vec's, _tensor etc, NOT sym, asym)
	static_assert(is_same_v<float3::RemoveIthNesting<0>, float>);
	static_assert(is_same_v<float3x3::RemoveIthNesting<0>, float3>);
	static_assert(is_same_v<float3x3::RemoveIthNesting<1>, float3>);
	
	static_assert(is_same_v<float3s3::RemoveIthNesting<0>, float>);
	static_assert(is_same_v<float3a3::RemoveIthNesting<0>, float>);
	static_assert(is_same_v<_tensor<int,2,3,4>::RemoveIthNesting<0>, _tensor<int,3,4>>);
	static_assert(is_same_v<_tensor<int,2,3,4>::RemoveIthNesting<1>, _tensor<int,2,4>>);
	static_assert(is_same_v<_tensor<int,2,3,4>::RemoveIthNesting<2>, _tensor<int,2,3>>);

	static_assert(is_same_v<float3::RemoveIndex<0>, float>);
	static_assert(is_same_v<float3x3::RemoveIndex<0>, float3>);
	static_assert(is_same_v<float3x3::RemoveIndex<1>, float3>);
	static_assert(is_same_v<float3s3::RemoveIndex<0>, float3>);
	static_assert(is_same_v<float3s3::RemoveIndex<1>, float3>);
	static_assert(is_same_v<float3a3::RemoveIndex<0>, float3>);
	static_assert(is_same_v<float3a3::RemoveIndex<1>, float3>);
	
	static_assert(is_same_v<_tensor<int,2,3,4>::RemoveIndex<0>, _tensor<int,3,4>>);
	static_assert(is_same_v<_tensor<int,2,3,4>::RemoveIndex<1>, _tensor<int,2,4>>);
	static_assert(is_same_v<_tensor<int,2,3,4>::RemoveIndex<2>, _tensor<int,2,3>>);

	// verify RemoveIndex<> is order indepenent
	//[a,b,c,d] remove 0 => [a,c,d] => remove 3 => compiler error
	//[a,b,c,d] remove 3 => [a,b,c] => remove 0 => [b,c]
	static_assert(
		is_same_v<
			_tensor<int,2,3,4,5>::RemoveIndex<3,0>,
			_tensor<int,3,4>
		>
	);
	static_assert(
		is_same_v<
			_tensor<int,2,3,4,5>::RemoveIndex<0,3>,
			_tensor<int,3,4>
		>
	);


	static_assert(
		float3s3::numNestingsToIndex<0> == float3s3::numNestingsToIndex<1>
	);
	
	static_assert(
		is_sym_v<float3s3>
	);

	namespace Test3 {
		using float3x3x3 = _tensorr<float, 3, 3>;
		using float3s3x3 = _tensori<float, index_sym<3>, index_vec<3>>;
		using float3a3x3 = _tensori<float, index_asym<3>, index_vec<3>>;
		using float3x3s3 = _tensori<float, index_vec<3>, index_sym<3>>;
		using float3x3a3 = _tensori<float, index_vec<3>, index_asym<3>>;
		
		using float3x3x3x3 = _tensorr<float, 3, 4>;
		using float3s3x3x3 = _tensori<float, index_sym<3>, index_vec<3>, index_vec<3>>;
		using float3a3x3x3 = _tensori<float, index_asym<3>, index_vec<3>, index_vec<3>>;
		using float3x3s3x3 = _tensori<float, index_vec<3>, index_sym<3>, index_vec<3>>;
		using float3x3a3x3 = _tensori<float, index_vec<3>, index_asym<3>, index_vec<3>>;
		using float3x3x3s3 = _tensori<float, index_vec<3>, index_vec<3>, index_sym<3>>;
		using float3x3x3a3 = _tensori<float, index_vec<3>, index_vec<3>, index_asym<3>>;
		using float3s3x3s3 = _tensori<float, index_sym<3>, index_sym<3>>;
		using float3a3x3s3 = _tensori<float, index_asym<3>, index_sym<3>>;
		using float3s3x3a3 = _tensori<float, index_sym<3>, index_asym<3>>;
		using float3a3x3a3 = _tensori<float, index_asym<3>, index_asym<3>>;

		static_assert(is_same_v<float3::InnerForIndex<0>, float3>);
		static_assert(is_same_v<float3::InnerForIndex<1>, float>);
		
		static_assert(is_same_v<float3x3::InnerForIndex<0>, float3x3>);
		static_assert(is_same_v<float3x3::InnerForIndex<1>, float3>);
		static_assert(is_same_v<float3x3::InnerForIndex<2>, float>);
		
		static_assert(is_same_v<float3s3::InnerForIndex<0>, float3s3>);
		static_assert(is_same_v<float3s3::InnerForIndex<1>, float3s3>);
		static_assert(is_same_v<float3s3::InnerForIndex<2>, float>);
		
		static_assert(is_same_v<float3a3::InnerForIndex<0>, float3a3>);
		static_assert(is_same_v<float3a3::InnerForIndex<1>, float3a3>);
		static_assert(is_same_v<float3a3::InnerForIndex<2>, float>);


		static_assert(is_same_v<float3x3x3::InnerForIndex<0>, float3x3x3>);
		static_assert(is_same_v<float3x3x3::InnerForIndex<1>, float3x3>);
		static_assert(is_same_v<float3x3x3::InnerForIndex<2>, float3>);
		static_assert(is_same_v<float3x3x3::InnerForIndex<3>, float>);
		
		static_assert(is_same_v<float3s3x3::InnerForIndex<0>, float3s3x3>);
		static_assert(is_same_v<float3s3x3::InnerForIndex<1>, float3s3x3>);
		static_assert(is_same_v<float3s3x3::InnerForIndex<2>, float3>);
		static_assert(is_same_v<float3s3x3::InnerForIndex<3>, float>);
		
		static_assert(is_same_v<float3a3x3::InnerForIndex<0>, float3a3x3>);
		static_assert(is_same_v<float3a3x3::InnerForIndex<1>, float3a3x3>);
		static_assert(is_same_v<float3a3x3::InnerForIndex<2>, float3>);
		static_assert(is_same_v<float3a3x3::InnerForIndex<3>, float>);
		
		static_assert(is_same_v<float3x3s3::InnerForIndex<0>, float3x3s3>);
		static_assert(is_same_v<float3x3s3::InnerForIndex<1>, float3s3>);
		static_assert(is_same_v<float3x3s3::InnerForIndex<2>, float3s3>);
		static_assert(is_same_v<float3x3s3::InnerForIndex<3>, float>);
		
		static_assert(is_same_v<float3x3a3::InnerForIndex<0>, float3x3a3>);
		static_assert(is_same_v<float3x3a3::InnerForIndex<1>, float3a3>);
		static_assert(is_same_v<float3x3a3::InnerForIndex<2>, float3a3>);
		static_assert(is_same_v<float3x3a3::InnerForIndex<3>, float>);
	
		static_assert(is_same_v<float3x3x3x3::InnerForIndex<0>, float3x3x3x3>);
		static_assert(is_same_v<float3x3x3x3::InnerForIndex<1>, float3x3x3>);
		static_assert(is_same_v<float3x3x3x3::InnerForIndex<2>, float3x3>);
		static_assert(is_same_v<float3x3x3x3::InnerForIndex<3>, float3>);
		static_assert(is_same_v<float3x3x3x3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3s3x3x3::InnerForIndex<0>, float3s3x3x3>);
		static_assert(is_same_v<float3s3x3x3::InnerForIndex<1>, float3s3x3x3>);
		static_assert(is_same_v<float3s3x3x3::InnerForIndex<2>, float3x3>);
		static_assert(is_same_v<float3s3x3x3::InnerForIndex<3>, float3>);
		static_assert(is_same_v<float3s3x3x3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3a3x3x3::InnerForIndex<0>, float3a3x3x3>);
		static_assert(is_same_v<float3a3x3x3::InnerForIndex<1>, float3a3x3x3>);
		static_assert(is_same_v<float3a3x3x3::InnerForIndex<2>, float3x3>);
		static_assert(is_same_v<float3a3x3x3::InnerForIndex<3>, float3>);
		static_assert(is_same_v<float3a3x3x3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3x3s3x3::InnerForIndex<0>, float3x3s3x3>);
		static_assert(is_same_v<float3x3s3x3::InnerForIndex<1>, float3s3x3>);
		static_assert(is_same_v<float3x3s3x3::InnerForIndex<2>, float3s3x3>);
		static_assert(is_same_v<float3x3s3x3::InnerForIndex<3>, float3>);
		static_assert(is_same_v<float3x3s3x3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3x3a3x3::InnerForIndex<0>, float3x3a3x3>);
		static_assert(is_same_v<float3x3a3x3::InnerForIndex<1>, float3a3x3>);
		static_assert(is_same_v<float3x3a3x3::InnerForIndex<2>, float3a3x3>);
		static_assert(is_same_v<float3x3a3x3::InnerForIndex<3>, float3>);
		static_assert(is_same_v<float3x3a3x3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3x3x3s3::InnerForIndex<0>, float3x3x3s3>);
		static_assert(is_same_v<float3x3x3s3::InnerForIndex<1>, float3x3s3>);
		static_assert(is_same_v<float3x3x3s3::InnerForIndex<2>, float3s3>);
		static_assert(is_same_v<float3x3x3s3::InnerForIndex<3>, float3s3>);
		static_assert(is_same_v<float3x3x3s3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3x3x3a3::InnerForIndex<0>, float3x3x3a3>);
		static_assert(is_same_v<float3x3x3a3::InnerForIndex<1>, float3x3a3>);
		static_assert(is_same_v<float3x3x3a3::InnerForIndex<2>, float3a3>);
		static_assert(is_same_v<float3x3x3a3::InnerForIndex<3>, float3a3>);
		static_assert(is_same_v<float3x3x3a3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3s3x3s3::InnerForIndex<0>, float3s3x3s3>);
		static_assert(is_same_v<float3s3x3s3::InnerForIndex<1>, float3s3x3s3>);
		static_assert(is_same_v<float3s3x3s3::InnerForIndex<2>, float3s3>);
		static_assert(is_same_v<float3s3x3s3::InnerForIndex<3>, float3s3>);
		static_assert(is_same_v<float3s3x3s3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3a3x3s3::InnerForIndex<0>, float3a3x3s3>);
		static_assert(is_same_v<float3a3x3s3::InnerForIndex<1>, float3a3x3s3>);
		static_assert(is_same_v<float3a3x3s3::InnerForIndex<2>, float3s3>);
		static_assert(is_same_v<float3a3x3s3::InnerForIndex<3>, float3s3>);
		static_assert(is_same_v<float3a3x3s3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3s3x3a3::InnerForIndex<0>, float3s3x3a3>);
		static_assert(is_same_v<float3s3x3a3::InnerForIndex<1>, float3s3x3a3>);
		static_assert(is_same_v<float3s3x3a3::InnerForIndex<2>, float3a3>);
		static_assert(is_same_v<float3s3x3a3::InnerForIndex<3>, float3a3>);
		static_assert(is_same_v<float3s3x3a3::InnerForIndex<4>, float>);
		static_assert(is_same_v<float3a3x3a3::InnerForIndex<0>, float3a3x3a3>);
		static_assert(is_same_v<float3a3x3a3::InnerForIndex<1>, float3a3x3a3>);
		static_assert(is_same_v<float3a3x3a3::InnerForIndex<2>, float3a3>);
		static_assert(is_same_v<float3a3x3a3::InnerForIndex<3>, float3a3>);
		static_assert(is_same_v<float3a3x3a3::InnerForIndex<4>, float>);
	}

	namespace transposeTest {
		using T = float2x3;
		constexpr int m = 0;
		constexpr int n = 1;
		using Emn = typename T::template ExpandIndex<m,n>;
		static_assert(is_same_v<Emn, float2x3>);
		constexpr int mdim = Emn::template dim<m>;
		static_assert(mdim == 2);
		constexpr int ndim = Emn::template dim<n>;
		static_assert(ndim == 3);
		
		static_assert(Emn::numNestingsToIndex<0> == 0); // m=0 from-index
		static_assert(Emn::numNestingsToIndex<1> == 1);
		static_assert(Emn::numNestingsToIndex<2> == 2);
		
		static_assert(is_same_v<Emn::InnerForIndex<0>, float2x3>); // m=0 from-index
		static_assert(is_same_v<Emn::InnerForIndex<1>, float3>);
		static_assert(is_same_v<Emn::InnerForIndex<2>, float>);

		static_assert(is_same_v<float2x3::ReplaceLocalDim<4>, float4x3>);
		static_assert(is_same_v<float2::ReplaceLocalDim<4>, float4>);

		using Enn = typename Emn::template ReplaceNested<
			Emn::template numNestingsToIndex<m>,
			typename Emn::template InnerForIndex<m>::template ReplaceLocalDim<ndim>
		>;
		static_assert(is_same_v<Enn, float3x3>);
	
		static_assert(Enn::template numNestingsToIndex<0> == 0);
		static_assert(Enn::template numNestingsToIndex<1> == 1); // n=1 to-index
		static_assert(Enn::template numNestingsToIndex<2> == 2);
		
		static_assert(is_same_v<Enn::InnerForIndex<0>, float3x3>);
		static_assert(is_same_v<Enn::InnerForIndex<1>, float3>); // n=1 from-index
		static_assert(is_same_v<Enn::InnerForIndex<2>, float>);
		
		static_assert(is_same_v<float3x3::ReplaceLocalDim<4>, float4x3>);
		static_assert(is_same_v<float3::ReplaceLocalDim<4>, float4>);
	
		using Enm = typename Enn::template ReplaceNested<
			Enn::template numNestingsToIndex<n>,
			typename Enn::template InnerForIndex<n>::template ReplaceLocalDim<mdim>
		>;

		static_assert(is_same_v<Enm, float3x2>);
	}
	static_assert(is_same_v<decltype(transpose(float3x3())),float3x3>);
	static_assert(is_same_v<decltype(transpose(float3s3())),float3s3>);
	static_assert(is_same_v<decltype(transpose(float3a3())),float3a3>);
	// test swapping dimensions correctly
	namespace transposeTest1 {
		using T = _tensor<int, 2,3,4,5>;
		static_assert(is_same_v<decltype(transpose<0,0>(T())), _tensor<int,2,3,4,5>>);
		static_assert(is_same_v<decltype(transpose<1,1>(T())), _tensor<int,2,3,4,5>>);
		static_assert(is_same_v<decltype(transpose<2,2>(T())), _tensor<int,2,3,4,5>>);
		static_assert(is_same_v<decltype(transpose<3,3>(T())), _tensor<int,2,3,4,5>>);
		static_assert(is_same_v<decltype(transpose<0,1>(T())), _tensor<int,3,2,4,5>>);
		static_assert(is_same_v<decltype(transpose<0,2>(T())), _tensor<int,4,3,2,5>>);
		static_assert(is_same_v<decltype(transpose<0,3>(T())), _tensor<int,5,3,4,2>>);
		static_assert(is_same_v<decltype(transpose<1,2>(T())), _tensor<int,2,4,3,5>>);
		static_assert(is_same_v<decltype(transpose<1,3>(T())), _tensor<int,2,5,4,3>>);
		static_assert(is_same_v<decltype(transpose<2,3>(T())), _tensor<int,2,3,5,4>>);
		static_assert(is_same_v<decltype(transpose<1,0>(T())), _tensor<int,3,2,4,5>>);
		static_assert(is_same_v<decltype(transpose<2,0>(T())), _tensor<int,4,3,2,5>>);
		static_assert(is_same_v<decltype(transpose<3,0>(T())), _tensor<int,5,3,4,2>>);
		static_assert(is_same_v<decltype(transpose<2,1>(T())), _tensor<int,2,4,3,5>>);
		static_assert(is_same_v<decltype(transpose<3,1>(T())), _tensor<int,2,5,4,3>>);
		static_assert(is_same_v<decltype(transpose<3,2>(T())), _tensor<int,2,3,5,4>>);
	}
	// test preserving storage
	namespace transposeTest2 {
		using T = _tensori<int, index_sym<3>, index_vec<3>>;
		using R = _tensorr<int, 3,3>;
		static_assert(is_same_v<decltype(transpose<0,0>(T())), T>);
		static_assert(is_same_v<decltype(transpose<1,1>(T())), T>);
		static_assert(is_same_v<decltype(transpose<2,2>(T())), T>);
		static_assert(is_same_v<decltype(transpose<0,1>(T())), T>);
		static_assert(is_same_v<decltype(transpose<0,2>(T())), R>);
		static_assert(is_same_v<decltype(transpose<1,2>(T())), R>);
		static_assert(is_same_v<decltype(transpose<1,0>(T())), T>);
		static_assert(is_same_v<decltype(transpose<2,0>(T())), R>);
		static_assert(is_same_v<decltype(transpose<2,1>(T())), R>);
	}
	namespace transposeTest4 {
		using T = _tensori<int, index_sym<3>, index_sym<4>>;
		static_assert(is_same_v<decltype(transpose<0,0>(T())), T>);
		static_assert(is_same_v<decltype(transpose<1,1>(T())), T>);
		static_assert(is_same_v<decltype(transpose<2,2>(T())), T>);
		static_assert(is_same_v<decltype(transpose<3,3>(T())), T>);
		static_assert(is_same_v<decltype(transpose<0,1>(T())), T>);
		static_assert(is_same_v<decltype(transpose<0,2>(T())), _tensor<int,4,3,3,4>>);
		static_assert(is_same_v<decltype(transpose<0,3>(T())), _tensor<int,4,3,4,3>>);
		static_assert(is_same_v<decltype(transpose<1,2>(T())), _tensor<int,3,4,3,4>>);
		static_assert(is_same_v<decltype(transpose<1,3>(T())), _tensor<int,3,4,4,3>>);
		static_assert(is_same_v<decltype(transpose<2,3>(T())), T>);
		static_assert(is_same_v<decltype(transpose<1,0>(T())), T>);
		static_assert(is_same_v<decltype(transpose<2,0>(T())), _tensor<int,4,3,3,4>>);
		static_assert(is_same_v<decltype(transpose<3,0>(T())), _tensor<int,4,3,4,3>>);
		static_assert(is_same_v<decltype(transpose<2,1>(T())), _tensor<int,3,4,3,4>>);
		static_assert(is_same_v<decltype(transpose<3,1>(T())), _tensor<int,3,4,4,3>>);
		static_assert(is_same_v<decltype(transpose<3,2>(T())), T>);
	}
}

namespace HasAccessorTest {
	static_assert(!Tensor::is_tensor_v<int>);
	static_assert(!Tensor::is_tensor_v<float>);
	static_assert(!Tensor::is_tensor_v<std::string>);
	static_assert(!Tensor::is_tensor_v<std::function<void()>>);
	static_assert(Tensor::is_tensor_v<Tensor::float3>);
	static_assert(Tensor::is_tensor_v<Tensor::float3x3>);
	static_assert(Tensor::is_tensor_v<Tensor::float3s3>);
	static_assert(Tensor::is_tensor_v<Tensor::float3a3>);
}

namespace Test3 {
	using namespace Tensor;
	using namespace std;

	using S = float;
	using S3 = _vec<S,3>;
	using S3x3 = _vec<S3,3>;
	using S3s3 = _sym<S,3>;
	using S3a3 = _asym<S,3>;
	using S3x3x3 = _vec<S3x3,3>;
	using S3x3s3 = _vec<S3s3,3>;
	using S3x3a3 = _vec<S3a3,3>;
	using S3s3x3 = _sym<S3,3>;
	using S3a3x3 = _asym<S3,3>;

	static_assert(!is_tensor_v<S>);
	static_assert(is_tensor_v<S3>);
	static_assert(is_tensor_v<S3x3>);
	static_assert(is_tensor_v<S3s3>);
	static_assert(is_tensor_v<S3x3x3>);
	static_assert(is_tensor_v<S3x3s3>);
	static_assert(is_tensor_v<S3s3x3>);

	static_assert(sizeof(S3) == sizeof(S) * 3);
	static_assert(sizeof(S3x3) == sizeof(S) * 3 * 3);
	static_assert(sizeof(S3s3) == sizeof(S) * 6);
	static_assert(sizeof(S3x3s3) == sizeof(S) * 3 * 6);
	static_assert(sizeof(S3x3x3) == sizeof(S) * 3 * 3 * 3);

	static_assert(is_same_v<S3::Inner, S>);
	static_assert(is_same_v<S3s3::Inner, S>);
	static_assert(is_same_v<S3x3::Inner, S3>);
	static_assert(is_same_v<S3x3s3::Inner, S3s3>);
	static_assert(is_same_v<S3s3x3::Inner, S3>);
	static_assert(is_same_v<S3x3x3::Inner, S3x3>);

	static_assert(is_same_v<S3::Scalar, S>);
	static_assert(is_same_v<S3s3::Scalar, S>);
	static_assert(is_same_v<S3x3::Scalar, S>);
	static_assert(is_same_v<S3x3s3::Scalar, S>);
	static_assert(is_same_v<S3s3x3::Scalar, S>);
	static_assert(is_same_v<S3x3x3::Scalar, S>);


// so more use cases ... for Scalar type S
//

	//operator[] and operator()
		
		// rank-1
	
	//_tensor<S, index_vec<3>>
	static_assert(is_same_v<decltype(S3()(0)), S&>);
	static_assert(is_same_v<decltype(S3()[0]), S&>);

		//rank-2
	
	//_tensor<S, index_vec<3>, index_vec<3>>
	static_assert(is_same_v<decltype(S3x3()(0)), S3&>);
	static_assert(is_same_v<decltype(S3x3()[0]), S3&>);
	static_assert(is_same_v<decltype(S3x3()(0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3()(0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3()[0](0)), S&>);
	static_assert(is_same_v<decltype(S3x3()(0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3()[0][0]), S&>);

	//_tensor<S, index_sym<3>>
	static_assert(is_same_v<decltype(S3s3()(0)), S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<decltype(S3s3()[0]), S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<decltype(S3s3()(0,0)), S&>);
	static_assert(is_same_v<decltype(S3s3()(0)(0)), S&>);
	static_assert(is_same_v<decltype(S3s3()[0](0)), S&>);
	static_assert(is_same_v<decltype(S3s3()(0)[0]), S&>);
	static_assert(is_same_v<decltype(S3s3()[0][0]), S&>);

	//_tensor<S, index_asym<3>>
	static_assert(is_same_v<decltype(S3a3()(0)), S3a3::Accessor<S3a3>>);
	static_assert(is_same_v<decltype(S3a3()[0]), S3a3::Accessor<S3a3>>);
	static_assert(is_same_v<decltype(S3a3()(0,0)), AntiSymRef<S>>);
	static_assert(is_same_v<decltype(S3a3()(0)(0)), AntiSymRef<S>>);
	static_assert(is_same_v<decltype(S3a3()[0](0)), AntiSymRef<S>>);
	static_assert(is_same_v<decltype(S3a3()(0)[0]), AntiSymRef<S>>);
	static_assert(is_same_v<decltype(S3a3()[0][0]), AntiSymRef<S>>);

		//rank-3
	
	//_tensor<S, index_vec<3>, index_vec<3>, index_vec<3>>
	static_assert(is_same_v<decltype(S3x3x3()(0)), S3x3&>);
	static_assert(is_same_v<decltype(S3x3x3()[0]), S3x3&>);
	static_assert(is_same_v<decltype(S3x3x3()(0)(0)), S3&>);
	static_assert(is_same_v<decltype(S3x3x3()(0,0)), S3&>);
	static_assert(is_same_v<decltype(S3x3x3()[0](0)), S3&>);
	static_assert(is_same_v<decltype(S3x3x3()(0)[0]), S3&>);
	static_assert(is_same_v<decltype(S3x3x3()[0][0]), S3&>);
	static_assert(is_same_v<decltype(S3x3x3()(0)(0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()(0,0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()(0)(0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()(0,0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()[0](0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()[0](0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()(0)[0](0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()(0)(0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3x3()(0,0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3x3()[0][0](0)), S&>);
	static_assert(is_same_v<decltype(S3x3x3()[0](0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3x3()(0)[0][0]), S&>);
	static_assert(is_same_v<decltype(S3x3x3()[0][0][0]), S&>);

	//_tensor<S, index_vec<3>, index_sym<3>>
	static_assert(is_same_v<decltype(S3x3s3()(0)), S3s3&>);
	static_assert(is_same_v<decltype(S3x3s3()[0]), S3s3&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)(0)), S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<decltype(S3x3s3()(0,0)), S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<decltype(S3x3s3()[0](0)), S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<decltype(S3x3s3()(0)[0]), S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<decltype(S3x3s3()[0][0]), S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<decltype(S3x3s3()(0)(0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0,0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)(0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0,0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0](0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0](0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)[0](0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)(0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0,0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0][0](0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0](0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)[0][0]), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0][0][0]), S&>);
}

namespace TestTotallySymmetric {
	using namespace Tensor;
	static_assert(factorial(0) == 1);
	static_assert(factorial(1) == 1);
	static_assert(factorial(2) == 2);
	static_assert(factorial(3) == 6);
	static_assert(factorial(4) == 24);

	static_assert(nChooseR(0,0) == 1);
	static_assert(nChooseR(1,0) == 1);
	static_assert(nChooseR(1,1) == 1);
	static_assert(nChooseR(2,0) == 1);
	static_assert(nChooseR(2,1) == 2);
	static_assert(nChooseR(2,2) == 1);
	static_assert(nChooseR(3,0) == 1);
	static_assert(nChooseR(3,1) == 3);
	static_assert(nChooseR(3,2) == 3);
	static_assert(nChooseR(3,3) == 1);
	static_assert(nChooseR(4,0) == 1);
	static_assert(nChooseR(4,1) == 4);
	static_assert(nChooseR(4,2) == 6);
	static_assert(nChooseR(4,3) == 4);
	static_assert(nChooseR(4,4) == 1);

	static_assert(sizeof(_symR<float, 1, 1>) == sizeof(float));
}

// more old tests
// TODO these are static_assert's
namespace StaticTest2 {
	using Real = double;
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<3>>::rank)== 1);
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<3>>::dim<0>)== 3);

	static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>>::rank)== 1);
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>>::dim<0>)== 4);

	static_assert((Tensor::_tensori<Real, Tensor::index_sym<3>>::rank)== 2);
	static_assert((Tensor::_tensori<Real, Tensor::index_sym<3>>::dim<0>)== 3);
	static_assert((Tensor::_tensori<Real, Tensor::index_sym<3>>::dim<1>)== 3);

	static_assert((Tensor::_tensori<Real, Tensor::index_vec<5>, Tensor::index_vec<6>>::rank)== 2);
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<5>, Tensor::index_vec<6>>::dim<0>)== 5);
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<5>, Tensor::index_vec<6>>::dim<1>)== 6);

	static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::rank)== 3);
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::dim<0>)== 4);
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::dim<1>)== 3);
	static_assert((Tensor::_tensori<Real, Tensor::index_vec<4>, Tensor::index_sym<3>>::dim<2>)== 3);

	static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::rank)== 4);
	static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<0>)== 2);
	static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<1>)== 2);
	static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<2>)== 3);
	static_assert((Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<3>>::dim<3>)== 3);
}

// verify that the outer of a vector and a sym is just that
namespace StaticTest3 {
	auto a = Tensor::float2(2,3);
	//ECHO(a);
	static_assert(a.numNestings == 1);
	static_assert(a.count<0> == 2);
	static_assert(a.rank == 1);
	static_assert(a.dim<0> == 2);
	auto b = Tensor::float3s3(6,5,4,3,2,1);
	//ECHO(b);
	static_assert(b.numNestings == 1);
	static_assert(b.count<0> == 6);
	static_assert(b.rank == 2);
	static_assert(b.dim<0> == 3);
	static_assert(b.dim<1> == 3);
	auto c = outer(a,b);
	//ECHO(c);
	static_assert(c.numNestings == 2);
	static_assert(c.count<0> == a.count<0>);
	static_assert(c.count<1> == b.count<0>);
	static_assert(c.rank == 3);
	static_assert(c.dim<0> == 2);
	static_assert(c.dim<1> == 3);
	static_assert(c.dim<2> == 3);
	auto d = outer(b,a);
	//ECHO(d);
	static_assert(d.numNestings == 2);
	static_assert(d.count<0> == b.count<0>);
	static_assert(d.count<1> == a.count<0>);
	static_assert(d.rank == 3);
	static_assert(d.dim<0> == 3);
	static_assert(d.dim<1> == 3);
	static_assert(d.dim<2> == 2);
}

namespace InteriorTest {
	using namespace Tensor;
	using namespace std;
	using A = float3x3;
	using B = float3x3;
	using R = typename A::template ReplaceScalar<B>;
	static_assert(is_same_v<_tensorr<float,3,4>, decltype(outer<A,B>(A(),B()))>);
	using R1 = typename A
		::template ReplaceScalar<B>
		::template RemoveIndex<A::rank-1, A::rank>;
	static_assert(is_same_v<R1, float3x3>);
	static_assert(is_same_v<R1, decltype(contract<A::rank-1,A::rank>(outer(A(),B())))>);
	using R2 = typename A
			::template ReplaceScalar<B>
			::template RemoveIndexSeq<Common::make_integer_range<int, A::rank-1, A::rank+1>>;
	static_assert(is_same_v<R1, R2>);
}


void test_TensorRank3() {

	// rank 3

	// rank-3 tensor: vector-vector-vector
	{
		using T = Tensor::_tensor<float, 2, 4, 5>;
		T t;
		static_assert(t.rank == 3);
		static_assert(t.dim<0> == 2);
		static_assert(t.dim<1> == 4);
		static_assert(t.dim<2> == 5);

		auto f = [](int i, int j, int k) -> float { return 3*i + 4*j*j + 5*k*k*k; };
		t = T(f);
		verifyAccessRank3<decltype(t)>(t,f);
		verifyAccessRank3<decltype(t) const>(t,f);
		
		// operators
		operatorScalarTest(t);

		//TODO tensor mul
	}

	// rank-3 tensor with intermixed non-vec types:
	// vector-of-symmetric
	{
		//this is a T_ijk = T_ikj, i spans 3 dims, j and k span 2 dims
		using T2s2x3 = Tensor::_tensori<float, Tensor::index_vec<2>, Tensor::index_sym<3>>;
		
		// list ctor
		T2s2x3 a = {
			{1,2,3,4,5,6}, //x_x x_y y_y x_z y_z z_z
			{7,8,9,0,1,2},
		};
		// xyz field access
		TEST_EQ(a.x.x_x, 1);
		TEST_EQ(a.x.x_y, 2);
		TEST_EQ(a.x.y_y, 3);
		TEST_EQ(a.x.x_z, 4);
		TEST_EQ(a.x.y_z, 5);
		TEST_EQ(a.x.z_z, 6);
		TEST_EQ(a.y.x_x, 7);
		TEST_EQ(a.y.x_y, 8);
		TEST_EQ(a.y.y_y, 9);
		TEST_EQ(a.y.x_z, 0);
		TEST_EQ(a.y.y_z, 1);
		TEST_EQ(a.y.z_z, 2);

		auto f = [](int i, int j, int k) -> float { return 4*i - j*j - k*k; };
		auto t = T2s2x3(f);
		verifyAccessRank3<decltype(t)>(t,f);
		verifyAccessRank3<decltype(t) const>(t,f);

		// operators
		operatorScalarTest(t);
		operatorMatrixTest<T2s2x3>();
	}

	//vector-of-symmetric
	{
		using T3x3s3 = Tensor::_tensori<float, Tensor::index_vec<3>, Tensor::index_sym<3>>;
		auto f = [](int i, int j, int k) -> float { return 4*i - j*j - k*k; };
		auto t = T3x3s3(f);
		verifyAccessRank3<decltype(t)>(t, f);
		verifyAccessRank3<decltype(t) const>(t, f);
	}

#if 0
	//vector-of-antisymmetric
	{
		using T3x3a3 = Tensor::_tensori<float, Tensor::index_vec<3>, Tensor::index_asym<3>>;
		auto f = [](int i, int j, int k) -> float { return sign(k-j)*(j+k+4*i); };
		auto t = T3x3a3(f);
		verifyAccessRank3<decltype(t) const>(t, f);
		verifyAccessRank3<decltype(t)>(t, f);
	}
#endif
	
	// symmetric-of-vector
	{
		using T3s3x3 = Tensor::_tensori<float, Tensor::index_sym<3>, Tensor::index_vec<3>>;
		auto f = [](int i, int j, int k) -> float { return 4*k - i*i - j*j; };
		auto t = T3s3x3(f);
		verifyAccessRank3<decltype(t)>(t, f);
		verifyAccessRank3<decltype(t) const>(t, f);
	}

#if 0	// TODO
	// antisymmetric-of-vector
	{
		using T3a3x3 = Tensor::_tensori<float, Tensor::index_asym<3>, Tensor::index_vec<3>>;
		auto f = [](int i, int j, int k) -> float { return 4*k - i*i + j*j; };
		auto t = T3a3x3(f);
		verifyAccessRank3<decltype(t)>(t, f);
		verifyAccessRank3<decltype(t) const>(t, f);
	}
#endif
}
