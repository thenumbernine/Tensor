#include "Tensor/Tensor.h"
#include "Tensor/Inverse.h"
#include "Common/Test.h"

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
dot, inner
lenSq
length
distance
normalize
elemMul, hadamard, matrixCompMult
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
	static_assert(is_same_v< _tensor<int,3,3>::ExpandIthIndex<0>, _tensorr<int,3,2>>);
	static_assert(is_same_v< _tensor<int,3,3>::ExpandIthIndex<1>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_sym<int,3>::ExpandIthIndex<0>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_sym<int,3>::ExpandIthIndex<1>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_asym<int,3>::ExpandIthIndex<0>, _tensorr<int,3,2>>);
	static_assert(is_same_v<_asym<int,3>::ExpandIthIndex<1>, _tensorr<int,3,2>>);
	static_assert(is_same_v< _tensor<int,3,3,3>::ExpandIthIndex<0>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensor<int,3,3,3>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensor<int,3,3,3>::ExpandIthIndex<2>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_sym<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_sym<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_sym<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_sym<3>,index_vec<3>>>);
	static_assert(is_same_v< _tensori<int,index_asym<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_asym<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_asym<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_asym<3>,index_vec<3>>>);
	static_assert(is_same_v< _tensori<int,index_vec<3>,index_sym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_sym<3>>>);
	static_assert(is_same_v< _tensori<int,index_vec<3>,index_sym<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_vec<3>,index_sym<3>>::ExpandIthIndex<2>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_vec<3>,index_asym<3>>::ExpandIthIndex<0>, _tensori<int,index_vec<3>,index_asym<3>>>);
	static_assert(is_same_v< _tensori<int,index_vec<3>,index_asym<3>>::ExpandIthIndex<1>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensori<int,index_vec<3>,index_asym<3>>::ExpandIthIndex<2>, _tensorr<int,3,3>>);
	static_assert(is_same_v< _tensor<int,3,3,3,3>::ExpandIthIndex<0>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensor<int,3,3,3,3>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensor<int,3,3,3,3>::ExpandIthIndex<2>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensor<int,3,3,3,3>::ExpandIthIndex<3>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v< _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<3>, _tensori<int,index_sym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v< _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<0>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<1>, _tensorr<int,3,4>>);
	static_assert(is_same_v< _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<2>, _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>>);
	static_assert(is_same_v< _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>::ExpandIthIndex<3>, _tensori<int,index_asym<3>,index_vec<3>,index_vec<3>>>);
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
	static_assert(
		is_same_v<
			decltype(
				transpose(float3x3())
			),
			float3x3
		>
	);
	static_assert(
		is_same_v<
			decltype(
				transpose(float3s3())
			),
			float3s3
		>
	);
#if 0 // TODO fixme by implementing _asym's operator(intN)	
	static_assert(
		is_same_v<
			decltype(
				transpose(float3a3())
			),
			float3x3
		>
	);
#endif	
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
	//static_assert(Tensor::is_tensor_v<Tensor::quatf>);	// TODO put this in the src/Quat.cpp test, or #include "Tensor/Quat.h" ... either way
	static_assert(!Tensor::has_Accessor_v<float>);
	static_assert(!Tensor::has_Accessor_v<Tensor::float3>);
	static_assert(!Tensor::has_Accessor_v<Tensor::float3x3>);
	static_assert(Tensor::has_Accessor_v<Tensor::float3s3>);
	static_assert(Tensor::has_Accessor_v<Tensor::float3a3>);
	
	static_assert(std::is_same_v<Tensor::float3::IndexResult, float&>);
	static_assert(std::is_same_v<Tensor::float3::IndexResultConst, float const&>);
	static_assert(std::is_same_v<Tensor::float3x3::IndexResult, float&>);
	static_assert(std::is_same_v<Tensor::float3x3::IndexResultConst, float const&>);
	static_assert(std::is_same_v<Tensor::float3x3::IndexResult, float&>);
	static_assert(std::is_same_v<Tensor::float3x3::IndexResultConst, float const&>);
	static_assert(std::is_same_v<Tensor::float3s3::IndexResult, Tensor::float3s3::Accessor<Tensor::float3s3>>);
	static_assert(std::is_same_v<Tensor::float3s3::IndexResultConst, Tensor::float3s3::Accessor<Tensor::float3s3 const>>);
	static_assert(std::is_same_v<Tensor::float3a3::IndexResult, Tensor::float3a3::Accessor<Tensor::float3a3>>);
	static_assert(std::is_same_v<Tensor::float3a3::IndexResultConst, Tensor::float3a3::Accessor<Tensor::float3a3 const>>);
}

//is adding "const" to "float&" the same as adding "&" to "float const" ?
// no.  it's not.  "float & const"  (which can't exist) vs "float const &"
static_assert(!std::is_same_v<typename Tensor::constness_of<float const>::template apply_t<float&>, std::add_lvalue_reference_t<float const>>);
// for pointers this should be false.
// "float * const" vs "float const *"
static_assert(!std::is_same_v<typename Tensor::constness_of<float const>::template apply_t<float*>, std::add_pointer_t<float const>>);

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

	// does it have an 'Accessor' member template
	static_assert(!has_Accessor_v<S>);
	static_assert(!has_Accessor_v<S3>);
	static_assert(has_Accessor_v<S3s3>);
	static_assert(!has_Accessor_v<S3x3>);
	static_assert(!has_Accessor_v<S3x3s3>);
	static_assert(has_Accessor_v<S3s3x3>);
	static_assert(!has_Accessor_v<S3x3x3>);


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

	// get the IndexResult i.e. result of indexing operations (call, subscript)
	//  if any template-nested classes have a member Accessor<This [const]> then use that
	//  otherwise use the Scalar [const] &
	static_assert(is_same_v<S3::IndexResult, S&>);
	static_assert(is_same_v<S3::IndexResultConst, S const &>);
	static_assert(is_same_v<S3s3::IndexResult, S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<S3s3::IndexResultConst, S3s3::Accessor<S3s3 const>>);
	static_assert(is_same_v<S3x3::IndexResult, S&>);
	static_assert(is_same_v<S3x3::IndexResultConst, S const &>);
	static_assert(is_same_v<S3x3s3::IndexResult, S3s3::Accessor<S3s3>>);
	static_assert(is_same_v<S3x3s3::IndexResultConst, S3s3::Accessor<S3s3 const>>);
	static_assert(is_same_v<S3s3x3::IndexResult, S3s3x3::Accessor<S3s3x3>>);
	static_assert(is_same_v<S3s3x3::IndexResultConst, S3s3x3::Accessor<S3s3x3 const>>);

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
	static_assert(is_same_v<decltype(S3s3()(0)), S3s3::IndexResult>);
	static_assert(is_same_v<decltype(S3s3()[0]), S3s3::IndexResult>);
	static_assert(is_same_v<decltype(S3s3()(0,0)), S&>);
	static_assert(is_same_v<decltype(S3s3()(0)(0)), S&>);
	static_assert(is_same_v<decltype(S3s3()[0](0)), S&>);
	static_assert(is_same_v<decltype(S3s3()(0)[0]), S&>);
	static_assert(is_same_v<decltype(S3s3()[0][0]), S&>);

	//_tensor<S, index_asym<3>>
	static_assert(is_same_v<decltype(S3a3()(0)), S3a3::IndexResult>);
	static_assert(is_same_v<decltype(S3a3()[0]), S3a3::IndexResult>);
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
	static_assert(is_same_v<decltype(S3x3s3()(0)(0)), S3s3::IndexResult>);

	//TODO
	//this requires _vec's operator(int, int...) to return its IndexResult
	// but which IndexResult? that depends on the # of (int...)'s 
	// so for (int, int), it should be This::InnerForIndex<2>::IndexResult
//	static_assert(is_same_v<decltype(S3x3s3()(0,0)), S3s3::IndexResult>);
	
	static_assert(is_same_v<decltype(S3x3s3()[0](0)), S3s3::IndexResult>);
	static_assert(is_same_v<decltype(S3x3s3()(0)[0]), S3s3::IndexResult>);
	static_assert(is_same_v<decltype(S3x3s3()[0][0]), S3s3::IndexResult>);
	static_assert(is_same_v<decltype(S3x3s3()(0)(0)(0)), S&>);
//	static_assert(is_same_v<decltype(S3x3s3()(0,0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)(0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0,0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0](0)(0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0](0,0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)[0](0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)(0)[0]), S&>);
//	static_assert(is_same_v<decltype(S3x3s3()(0,0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0][0](0)), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0](0)[0]), S&>);
	static_assert(is_same_v<decltype(S3x3s3()(0)[0][0]), S&>);
	static_assert(is_same_v<decltype(S3x3s3()[0][0][0]), S&>);



}

template<typename T>
void operatorScalarTest(T const & t) {
	using S = typename T::Scalar;
	TEST_EQ(t + (S)0, t);
	TEST_EQ((S)1 + t, t + T((S)1));
	TEST_EQ(t + T(), t);
	TEST_EQ(t - T(), t);
	TEST_EQ((S)0 - t, -t);
	TEST_EQ(t - (S)0, t);
	TEST_EQ(t - (S)1, t - T((S)1));
	TEST_EQ(t - t, T());
	TEST_EQ(t - t * (S)2, -t);
	TEST_EQ(t - (S)2 * t, -t);
	TEST_EQ(t * (S)1, t);
	TEST_EQ(t * (S)-1, -t);
	TEST_EQ((S)-1 * t, -t);
	TEST_EQ(t * (S)2, t + t);
	TEST_EQ(t * (S)0, T());
	TEST_EQ(t / (S)1, t);
	TEST_EQ(t / (S).5, (S)2 * t);
	//TEST_EQ(t / t, T((S)1)); // if t(I) == 0 then this gives nan ... so ... 
	TEST_EQ(t / T((S)1), t);
}

template<typename T>
void operatorMatrixTest() {
	static_assert(T::rank >= 2);
}

void test_Tensor() {
	//vector
	
	{
		// default ctor
		Tensor::float3 f;
		for (int i = 0; i < f.dim<0>; ++i) {
			TEST_EQ(f.s[i], 0);
		}
	}

	{
		// parenthesis ctor
		Tensor::float3 f(4,5,7);
		
		// initializer list ctor
		Tensor::float3 g = {7,1,2};
		
		//.dims
		static_assert(f.rank == 1);
		static_assert(f.dims == 3);
		static_assert(f.dim<0> == 3);
		static_assert(f.numNestings == 1);
		static_assert(f.count<0> == 3);
	
		//test .x .y .z
		TEST_EQ(f.x, 4);
		TEST_EQ(f.y, 5);
		TEST_EQ(f.z, 7);
		// test .s0 .s1 .s2
		TEST_EQ(f.s0, 4);
		TEST_EQ(f.s1, 5);
		TEST_EQ(f.s2, 7);
		// test .s[]
		TEST_EQ(f.s[0], 4);
		TEST_EQ(f.s[1], 5);
		TEST_EQ(f.s[2], 7);
		// test () indexing
		TEST_EQ(f(0), 4);
		TEST_EQ(f(1), 5);
		TEST_EQ(f(2), 7);
		
		//test [] indexing
		TEST_EQ(f[0], 4);
		TEST_EQ(f[1], 5);
		TEST_EQ(f[2], 7);

		// indexing
		{
			auto f = [](int i) -> float { return i+1; };
			auto verifyAccess = []<typename T, typename F>(T & t, F f){
				for (int i = 0; i < T::template dim<0>; ++i) {
					typename T::Scalar e = f(i);
					// various [] and (int...) and (intN)
					TEST_EQ(t(i), e);
					TEST_EQ(t(Tensor::intN<1>(i)), e);
					TEST_EQ(t[i], e);
					TEST_EQ(t.s[i], e);
				}
			};
			Tensor::float3 t(f);
			verifyAccess.template operator()<decltype(t)>(t, f);
			verifyAccess.template operator()<decltype(t) const>(t, f);
		}

		//lambda ctor
		TEST_EQ(f, Tensor::float3([](int i) -> float { return 4 + i * (i + 1) / 2; }));
		TEST_EQ(f, Tensor::float3([](Tensor::intN<1> i) -> float { return 4 + i(0) * (i(0) + 1) / 2; }));

		// scalar ctor
		TEST_EQ(Tensor::float3(3), Tensor::float3(3,3,3));

		// casting
		Tensor::int3 fi = {4,5,7};
		Tensor::double3 fd = {4,5,7};
		TEST_EQ(f, (Tensor::float3)fi);
		TEST_EQ(f, (Tensor::float3)fd);

		//iterator
		{
			auto i = f.begin();
			TEST_EQ(*i, 4); ++i;
			TEST_EQ(*i, 5); i++;
			TEST_EQ(*i, 7); i++;
			TEST_EQ(i, f.end());
		
			for (auto & i : f) {
				std::cout << "f iter = " << i << std::endl;
			}
			for (auto const & i : f) {
				std::cout << "f iter = " << i << std::endl;
			}
			// TODO verify cbegin/cend
			// TODO support for rbegin/rend const/not const and crbegin/crend
		}
	
		// operators 
		// vector/scalar operations
		TEST_EQ(f+1.f, Tensor::float3(5,6,8));
		TEST_EQ(f-1.f, Tensor::float3(3,4,6));
		TEST_EQ(f*12.f, Tensor::float3(48, 60, 84));
		TEST_EQ(f/2.f, Tensor::float3(2.f, 2.5f, 3.5f));
		// scalar/vector operations
		TEST_EQ(1.f+f, Tensor::float3(5,6,8));
		TEST_EQ(1.f-f, Tensor::float3(-3, -4, -6));
		TEST_EQ(12.f*f, Tensor::float3(48, 60, 84));
		TEST_EQ(2.f/f, Tensor::float3(0.5, 0.4, 0.28571428571429));
		// vector/vector operations
		TEST_EQ(f+g, Tensor::float3(11, 6, 9));
		TEST_EQ(f-g, Tensor::float3(-3, 4, 5));
		TEST_EQ(f/g, Tensor::float3(0.57142857142857, 5.0, 3.5));	// wow, this equality passes while sqrt(90) fails
	
		// for vector*vector I'm picking the scalar-result of the 2 GLSL options (are there three? can you do mat = vec * vec in GLSL?)
		//  this fits with general compatability of tensor operator* being outer+contract
		TEST_EQ(f*g, 47)

		// op= scalar
		{ Tensor::float3 h = f; h += 2; TEST_EQ(h, Tensor::float3(6,7,9)); }
		{ Tensor::float3 h = f; h -= 3; TEST_EQ(h, Tensor::float3(1,2,4)); }
		{ Tensor::float3 h = f; h *= 3; TEST_EQ(h, Tensor::float3(12,15,21)); }
		{ Tensor::float3 h = f; h /= 4; TEST_EQ(h, Tensor::float3(1,1.25,1.75)); }
		// op= vector
		{ Tensor::float3 h = f; h += Tensor::float3(3,2,1); TEST_EQ(h, Tensor::float3(7,7,8)); }
		{ Tensor::float3 h = f; h -= Tensor::float3(5,0,9); TEST_EQ(h, Tensor::float3(-1,5,-2)); }
		{ Tensor::float3 h = f; h *= Tensor::float3(-1,1,-2); TEST_EQ(h, Tensor::float3(-4,5,-14)); }
		{ Tensor::float3 h = f; h /= Tensor::float3(-2,3,-4); TEST_EQ(h, Tensor::float3(-2, 5.f/3.f, -1.75)); }

		//operator<< works?
		std::cout << f << std::endl;
		// to_string works?
		std::cout << std::to_string(f) << std::endl;

		// dot product
		TEST_EQ(dot(f,g), 47)
		TEST_EQ(f.dot(g), 47)
		
		// length-squared
		TEST_EQ(f.lenSq(), 90);
		TEST_EQ(lenSq(f), 90);

		// length
		TEST_EQ_EPS(f.length(), sqrt(90), 1e-6);
		TEST_EQ_EPS(length(f), sqrt(90), 1e-6);
		
		// cros product
		TEST_EQ(cross(f,g), Tensor::float3(3, 41, -31))

		// outer product
		// hmm, in the old days macros couldn't detect <>'s so you'd have to wrap them in ()'s if the <>'s had ,'s in them
		// now same persists for {}'s it seems
		auto fouterg = outer(f,g);
		TEST_EQ(fouterg, Tensor::float3x3(
			{28, 4, 8},
			{35, 5, 10},
			{49, 7, 14}
		));

		auto gouterf = transpose(fouterg);
		TEST_EQ(gouterf, Tensor::float3x3(
			{28, 35, 49},
			{4, 5, 7},
			{8, 10, 14}
		));

		// TODO vector subset access
	
		// swizzle
		// TODO need an operator== between T and reference_wrapper<T> ...
		// or casting ctor?
		// a generic ctor between _vecs would be nice, but maybe problematic for _mat = _sym
		TEST_EQ(Tensor::float3(f.zyx()), Tensor::float3(7,5,4));
		TEST_EQ(Tensor::float2(f.xy()), Tensor::float2(4,5));
		TEST_EQ(Tensor::float2(f.yx()), Tensor::float2(5,4));
		TEST_EQ(Tensor::float2(f.yy()), Tensor::float2(5,5));
		
		/* more tests ...
		float2 float4
		int2 int3 int4
		
		default-template vectors of dif sizes (5 maybe? ... )
			assert that no .x exists to verify
		*/

		// verify vec4 list constructor works
		Tensor::float4 h = {2,3,4,5};
		TEST_EQ(h.x, 2);
		TEST_EQ(h.y, 3);
		TEST_EQ(h.z, 4);
		TEST_EQ(h.w, 5);

		Tensor::_vec<float,5> j = {5,6,7,8,9};
		//non-specialized: can't use xyzw for dim>4
		TEST_EQ(j[0], 5);
		TEST_EQ(j[1], 6);
		TEST_EQ(j[2], 7);
		TEST_EQ(j[3], 8);
		TEST_EQ(j[4], 9);
	
		// iterator copy
		Tensor::float3 f2;
		std::copy(f2.begin(), f2.end(), f.begin()); // crashing ...
		TEST_EQ(f, f2);
	
		// iterator copy from somewhere else
		std::array<float, 3> fa;
		std::copy(fa.begin(), fa.end(), f.begin());
		TEST_EQ(fa[0], f[0]);
		TEST_EQ(fa[1], f[1]);
		TEST_EQ(fa[2], f[2]);
	
		// operators
		operatorScalarTest(f);
	
		// contract

		TEST_EQ((Tensor::contract<0,0>(Tensor::float3(1,2,3))), 6);
	}

	//old libraries' tests
	{
		{
			//arg ctor works
			Tensor::float3 a(1,2,3);
			
			//bracket ctor works
			Tensor::float3 b = {4,5,6};

			//access
			TEST_EQ(a(0), 1);
			TEST_EQ(a[0], 1);

			//make sure GenericArray functionality works
			TEST_EQ(Tensor::float3(1), Tensor::float3(1,1,1));
			
			// new lib doesn't support this ... but should it?
			//TEST_EQ(Tensor::float3(1,2), Tensor::float3(1,2,0));
			
			TEST_EQ(b + a, Tensor::float3(5,7,9));
			TEST_EQ(b - a, Tensor::float3(3,3,3));
			TEST_EQ(b * a, 32);
			TEST_EQ(Tensor::float3(2,4,6)/Tensor::float3(1,2,3), Tensor::float3(2,2,2));
			TEST_EQ(b * 2., Tensor::float3(8, 10, 12));
			TEST_EQ(Tensor::float3(2,4,6)/2., Tensor::float3(1,2,3));	
		}
	}
	
	// rank-2

	auto verifyAccessRank2 = []<typename T, typename F>(T & t, F f){
		for (int i = 0; i < T::template dim<0>; ++i) {
			for (int j = 0; j < T::template dim<1>; ++j) {
				typename T::Scalar x = f(i,j);
				TEST_EQ(t(i)(j), x);
				TEST_EQ(t(i,j), x);
				// can't compile for _asym
				//TEST_EQ(t(Tensor::int2(i,j)), x);
				TEST_EQ(t[i](j), x);
				TEST_EQ(t(i)[j], x);
				TEST_EQ(t[i][j], x);
			}
		}
	};

	// matrix
	{
		//bracket ctor
		Tensor::float3x3 m = {
			{1,2,3},
			{4,5,6},
			{7,8,9},
		};
		
		//dims and rank.  really these are static_assert's, except dims, but it could be, but I'd have to constexpr some things ...
		static_assert(m.rank == 2);
		static_assert(m.dim<0> == 3);
		static_assert(m.dim<1> == 3);
		TEST_EQ(m.dims, Tensor::int2(3,3));
		static_assert(m.numNestings == 2);
		static_assert(m.count<0> == 3);
		static_assert(m.count<1> == 3);

		// .x .y .z indexing
		TEST_EQ(m.x.x, 1);
		TEST_EQ(m.x.y, 2);
		TEST_EQ(m.x.z, 3);
		TEST_EQ(m.y.x, 4);
		TEST_EQ(m.y.y, 5);
		TEST_EQ(m.y.z, 6);
		TEST_EQ(m.z.x, 7);
		TEST_EQ(m.z.y, 8);
		TEST_EQ(m.z.z, 9);

		// .s0 .s1 .s2 indexing
		TEST_EQ(m.s0.s0, 1);
		TEST_EQ(m.s0.s1, 2);
		TEST_EQ(m.s0.s2, 3);
		TEST_EQ(m.s1.s0, 4);
		TEST_EQ(m.s1.s1, 5);
		TEST_EQ(m.s1.s2, 6);
		TEST_EQ(m.s2.s0, 7);
		TEST_EQ(m.s2.s1, 8);
		TEST_EQ(m.s2.s2, 9);
		
		// indexing - various [] and (int...) and (intN)
		auto f = [](int i, int j) -> float { return 1 + j + 3 * i; };
		verifyAccessRank2.template operator()<decltype(m)>(m, f);
		verifyAccessRank2.template operator()<decltype(m) const>(m, f);
	
		//matrix-specific access , doesn't work for sym or asym
		auto verifyAccessMat = []<typename T, typename F>(T & t, F f) {
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					typename T::Scalar e = f(i,j);
					TEST_EQ(t.s[i](j), e);
					TEST_EQ(t.s[i][j], e);
					TEST_EQ(t(i).s[j], e);
					TEST_EQ(t[i].s[j], e);
					TEST_EQ(t.s[i].s[j], e);
				}
			}
		};
		verifyAccessMat.template operator()<decltype(m)>(m, f);
		verifyAccessMat.template operator()<decltype(m) const>(m, f);

		// scalar ctor
		// TODO how do GLSL matrix ctor from scalars work? 
		// do they initialize to full scalars like vecs do?
		// do they initialize to ident times scalar like math do?
		TEST_EQ(Tensor::float3x3(3), (Tensor::float3x3{{3,3,3},{3,3,3},{3,3,3}}));

		// lambda constructor
		// row-major, sequential in memory:
		TEST_EQ(m, Tensor::float3x3([](Tensor::int2 i) -> float { return 1 + i(1) + 3 * i(0); }));
		// col-major, sequential in memory:
		//TEST_EQ(m, Tensor::float3x3([](Tensor::int2 i) -> float { return 1 + i(0) + 3 * i(1); }));
		
		
		// TODO casting ctor

		// read iterator
		{
			auto i = m.begin();
			if constexpr (std::is_same_v<Tensor::int2::ReadInc<0>, Tensor::int2::ReadIncOuter<0>>) {
				// iterating in memory order for row-major
				// also in left-right top-bottom order when read
				// but you have to increment the last index first and first index last
				// TODO really?
				// but lambda init is now m(i,j) == 1 + i(1) + 3 * i(0) ... transposed of typical memory indexing
				TEST_EQ(*i, 1); ++i;
				TEST_EQ(*i, 2); ++i;
				TEST_EQ(*i, 3); ++i;
				TEST_EQ(*i, 4); ++i;
				TEST_EQ(*i, 5); ++i;
				TEST_EQ(*i, 6); ++i;
				TEST_EQ(*i, 7); ++i;
				TEST_EQ(*i, 8); ++i;
				TEST_EQ(*i, 9); ++i;
				TEST_EQ(i, m.end());
			} else if constexpr (std::is_same_v<Tensor::int2::ReadInc<0>, Tensor::int2::ReadIncInner<0>>) {
				//iterating transpose to memory order for row-major
				// - inc first index first, last index last
				// but lambda init is now m(i,j) == 1 + i(0) + 3 * i(1)  ... typical memory indexing
				// I could fulfill both at the same time by making my matrices column-major, like OpenGL does ... tempting ...
				TEST_EQ(*i, 1); ++i;
				TEST_EQ(*i, 4); ++i;
				TEST_EQ(*i, 7); ++i;
				TEST_EQ(*i, 2); ++i;
				TEST_EQ(*i, 5); ++i;
				TEST_EQ(*i, 8); ++i;
				TEST_EQ(*i, 3); ++i;
				TEST_EQ(*i, 6); ++i;
				TEST_EQ(*i, 9); ++i;
				TEST_EQ(i, m.end());
			}
		}
	
		// write iterator (should match read iterator except for symmetric members)

		// sub-vectors (row)
		TEST_EQ(m[0], Tensor::float3(1,2,3));
		TEST_EQ(m[1], Tensor::float3(4,5,6));
		TEST_EQ(m[2], Tensor::float3(7,8,9));

		// TODO matrix subset access

		// TODO matrix swizzle
	
		// operators
		operatorScalarTest(m);

		// operator *
		{
			auto a = Tensor::int2{7, -2};
			TEST_EQ((Tensor::contract<0,0>(a)), 5);
			auto b = Tensor::int2x2
				{{6, 9},
				{6, -6}};
			TEST_EQ((Tensor::contract<0,1>(b)), 0);
			TEST_EQ((Tensor::contract<1,0>(b)), 0);
			static_assert(std::is_same_v<Tensor::int2x2::RemoveIndex<0>, Tensor::int2>);
			ECHO((Tensor::contract<0,0>(b)));
			ECHO((Tensor::contract<1,1>(b)));

			auto aouterb = Tensor::_tensorr<int,2,3>{{{42, 63}, {42, -42}}, {{-12, -18}, {-12, 12}}};
			TEST_EQ(outer(a,b), aouterb);
			ECHO((Tensor::contract<0,0>(aouterb)));
			
			static_assert(std::is_same_v<Tensor::_tensorr<int,2,3>::RemoveIndex<0,1>, Tensor::int2>);
			ECHO((Tensor::contract<0,1>(aouterb)));
			
			ECHO((Tensor::contract<0,2>(aouterb)));
			ECHO((Tensor::contract<1,0>(aouterb)));
			ECHO((Tensor::contract<1,1>(aouterb)));
			ECHO((Tensor::contract<1,2>(aouterb)));
			ECHO((Tensor::contract<2,0>(aouterb)));
			ECHO((Tensor::contract<2,1>(aouterb)));
			ECHO((Tensor::contract<2,2>(aouterb)));
			auto atimesb = Tensor::int2{30, 75};
			TEST_EQ(a * b, atimesb);
			
			TEST_EQ( (Tensor::int2{-3, 6}
				* Tensor::int2x3{
					{-5, 0, -3},
					{1, 3, 0}}),
				(Tensor::int3{21, 18, 9}))
			TEST_EQ( (Tensor::int2{9, 9}
				* Tensor::int2x4{
					{3, -8, -10, -8},
					{5, 2, -5, 6}}),
				(Tensor::int4{72, -54, -135, -18}))
			TEST_EQ( (Tensor::int3{-7, -2, -8}
				* Tensor::int3x2{
					{-8, 0},
					{10, 7},
					{-6, 2}}),
				(Tensor::int2{84, -30}))
			TEST_EQ( (Tensor::int3{-4, 3, 1}
				* Tensor::int3x3{
					{0, 6, -2},
					{10, 1, 8},
					{-4, 6, -5}}),
				(Tensor::int3{26, -15, 27}))
			TEST_EQ( (Tensor::int3{-3, 6, 9}
				* Tensor::int3x4{
					{-9, -9, 8, -10},
					{9, -6, -3, -1},
					{1, 3, -9, -9}}),
				(Tensor::int4{90, 18, -123, -57}))
			TEST_EQ( (Tensor::int4{-5, 10, 8, 7}
				* Tensor::int4x2{
					{-5, 0},
					{1, 4},
					{-3, 1},
					{5, -10}}),
				(Tensor::int2{46, -22}))
			TEST_EQ( (Tensor::int4{-1, 9, 9, 5}
				* Tensor::int4x3{
					{-5, 4, 7},
					{5, -7, -4},
					{3, -1, -6},
					{-3, 8, 8}}),
				(Tensor::int3{62, -36, -57}))
			TEST_EQ( (Tensor::int4{-3, 4, 10, 2}
				* Tensor::int4x4{
					{3, -2, 0, -7},
					{8, 7, -6, 8},
					{-1, 4, 9, 3},
					{9, 9, 9, -1}}),
				(Tensor::int4{31, 92, 84, 81}))
		}

		// TODO make sure operator* matrix/vector, matrix/matrix, vector/matrix works
		// TODO I don't think I have marix *= working yet	

		auto m2 = elemMul(m,m);
		for (int i = 0; i < m.dim<0>; ++i) {
			for (int j = 0; j < m.dim<1>; ++j) {
				TEST_EQ(m2(i,j), m(i,j) * m(i,j));
			}
		}
		
		//determinant
		
		TEST_EQ(determinant(m), 0);

		// transpose

		TEST_EQ(Tensor::float3x3(
			{1,2,3},
			{4,5,6},
			{7,8,9}
		), Tensor::transpose(Tensor::float3x3(
			{1,4,7},
			{2,5,8},
			{3,6,9}
		)));
	
		TEST_EQ(Tensor::trace(Tensor::float3x3(
			{1,2,3},
			{4,5,6},
			{7,8,9}
		)), 15);
	}

	//symmetric
	
	{
		auto a = Tensor::float3s3(); // default
		static_assert(a.rank == 2);
		static_assert(a.dim<0> == 3);
		static_assert(a.dim<1> == 3);
		// default ctor
		for (int i = 0; i < a.count<0>; ++i) {
			TEST_EQ(a.s[i], 0);
		}
		for (int i = 0; i < a.dim<0>; ++i) {
			for (int j = 0; j < a.dim<1>; ++j) {
				TEST_EQ(a(i,j), 0);
				TEST_EQ(a(j,i), 0);
			}
		}
/*
use a symmetric procedural matrix with distinct values , esp for verifying .x_x fields
don't use a_ij = i+j, because a_02 == a_11
so here's a procedural symmetric matrix with all distinct symmetric components:
a_ij = i*i + j*j
	{{0, 1, 4},
	{1, 2, 5},
	{4, 5, 8}}
so a.s == {0,1,2,4,5,8};
*/
		a = Tensor::float3s3(0,1,2,4,5,8);
		
		// verify index access works

		auto verifyAccessSym = []<typename T>(T & a){	
			// testing fields 
			TEST_EQ(a.x_x, 0);
			TEST_EQ(a.x_y, 1);
			TEST_EQ(a.x_z, 4);
			TEST_EQ(a.y_x, 1);
			TEST_EQ(a.y_y, 2);
			TEST_EQ(a.y_z, 5);
			TEST_EQ(a.z_x, 4);
			TEST_EQ(a.z_y, 5);
			TEST_EQ(a.z_z, 8);
		};
		verifyAccessSym.template operator()<decltype(a)>(a);
		verifyAccessSym.template operator()<decltype(a) const>(a);
		
		auto f = [](int i, int j) -> float { return i*i + j*j; };
		verifyAccessRank2.template operator()<decltype(a)>(a, f);
		verifyAccessRank2.template operator()<decltype(a) const>(a, f);

		// lambda ctor using int,int...
		TEST_EQ(a, Tensor::float3s3([](int i, int j) -> float { return i*i + j*j; }));
		
		// lambda ctor using int2
		TEST_EQ(a, Tensor::float3s3([](Tensor::int2 ij) -> float { return ij(0)*ij(0) + ij(1)*ij(1); }));
		
		// verify only 6 writes take place during ctor
		{
			int k = 0;
			// verifies lambda-of-ref too
			auto c = Tensor::float3s3([&](int i, int j) -> float {
				++k;
				return (float)(i*i+j*j);
			});
			TEST_EQ(k, 6);	//for write iterators in lambda ctor ...
			TEST_EQ(c, a);
		}

		// lambda ctor using int2
		TEST_EQ(a, Tensor::float3s3([](Tensor::int2 ij) -> float {
			return (float)(ij.x*ij.x+ij.y*ij.y);
		}));


		/*
		test storing matrix
		for this test, construct from an asymmetric matrix
			0 1 2
			3 4 5
			6 7 8
		... in a symmetric tensor
		if storage / write iterate is lower-triangular then this will be 
		   0 3 6
		   3 4 7
		   6 7 8
		if it is upper-triangular:
			0 1 2
			1 4 5
			2 5 8
		*/
		auto b = Tensor::float3s3([](int i, int j) -> float {
			return 3 * i + j;
		});
#if 1 // upper triangular
		// test storage order
		// this order is for sym/asym getLocalReadForWriteIndex incrementing iread(0) first
		// it also means for _asym that i<j <=> POSITIVE, j<i <=> NEGATIVE
		TEST_EQ(b.s[0], 0);	// xx
		TEST_EQ(b.s[1], 3); // xy
		TEST_EQ(b.s[2], 4); // yy
		TEST_EQ(b.s[3], 6); // xz
		TEST_EQ(b.s[4], 7); // yz
		TEST_EQ(b.s[5], 8); // zz
		// test arg ctor 
		TEST_EQ(b, Tensor::float3s3(0,3,4,6,7,8));
#else // lower triangular
		// test storage order
		// this order is for sym/asym getLocalReadForWriteIndex incrementing iread(1) first
		// it also means for _asym that i<j <=> NEGATIVE, j<i <=> POSITIVE
		TEST_EQ(b.s[0], 0);	// xx
		TEST_EQ(b.s[1], 1); // xy
		TEST_EQ(b.s[2], 4); // yy
		TEST_EQ(b.s[3], 2); // xz
		TEST_EQ(b.s[4], 5); // yz
		TEST_EQ(b.s[5], 8); // zz
		// test arg ctor 
		TEST_EQ(b, Tensor::float3s3(0,1,4,2,5,8));
#endif

		// test symmetry read/write
		b(0,2) = 7;
		TEST_EQ(b(2,0), 7);
		b.x_y = -1;
		TEST_EQ(b.y_x, -1);

		// partial index
		for (int i = 0; i < b.dim<0>; ++i) {
			for (int j = 0; j < b.dim<1>; ++j) {
				TEST_EQ(b[i][j], b(i,j));
			}
		}
		
		// operators
		operatorScalarTest(a);
		operatorMatrixTest<Tensor::float3s3>();
	
		TEST_EQ(Tensor::trace(Tensor::float3s3({1,2,3,4,5,6})), 10);
	}

	// antisymmetric matrix
	{
		/*
		[ 0  1  2]
		[-1  0  3]
		[-2 -3  0]
		*/
		auto t = Tensor::float3a3{
			/*x_y=*/1,
			/*x_z=*/2,
			/*y_z=*/3
		};
		// (int,int) access
		t(0,0) = 1; //cannot write to diagonals
		t(1,1) = 2;
		t(2,2) = 3;

		// TODO indexes are 1 off
		auto a = Tensor::float3a3([](int i, int j) -> float { return i + j; });
		ECHO(a);
		TEST_EQ(a, t);

		auto b = Tensor::float3a3([](Tensor::int2 ij) -> float { return ij.x + ij.y; });
		ECHO(b);
		TEST_EQ(b, t);

		auto sign = [](int x) { return x == 0 ? 0 : (x < 0 ? -1 : 1); };
		auto f = [sign](int i, int j) -> float { return sign(j-i)*(i+j); };
		
		auto verifyAccessAntisym = []<typename T>(T & t){
			// "field" method access
			TEST_EQ(t.x_x(), 0);
			TEST_EQ(t.x_y(), 1);
			TEST_EQ(t.x_z(), 2);
			TEST_EQ(t.y_x(), -1);
			TEST_EQ(t.y_y(), 0);
			TEST_EQ(t.y_z(), 3);
			TEST_EQ(t.z_x(), -2);
			TEST_EQ(t.z_y(), -3);
			TEST_EQ(t.z_z(), 0);
		};
		verifyAccessAntisym.template operator()<decltype(t)>(t);
		verifyAccessAntisym.template operator()<decltype(t) const>(t);
		
		verifyAccessRank2.template operator()<decltype(t)>(t, f);
		verifyAccessRank2.template operator()<decltype(t) const>(t, f);

		// verify antisymmetric writes work
		for (int i = 0; i < t.dim<0>; ++i) {
			for (int j = 0; j < t.dim<1>; ++j) {
				float k = 1 + i + j;
				t(i,j) = k;
				if (i != j) {
					TEST_EQ(t(i,j), k);
					TEST_EQ(t(j,i), -k);
				} else {
					TEST_EQ(t(i,j), 0);
				}
			}
		}

		// TODO verify that 'float3a3::ExpandStorage<0> == float3x3' & same with <1> 

		// verify assignment to expanded type
		// TODO won't work until you get intN dereference in _asym
		Tensor::float3x3 c = t;
		TEST_EQ(c, (Tensor::float3x3{{0, -2, -3}, {2, 0, -4}, {3, 4, 0}}));
	
		//can't do yet until I fix asym access
		//operatorScalarTest(t);
		operatorMatrixTest<Tensor::float3a3>();
	}

	// rank-3 tensor: vector-vector-vector
	{
		using T = Tensor::_tensor<float, 2, 4, 5>;
		T t;
		static_assert(t.rank == 3);
		static_assert(t.dim<0> == 2);
		static_assert(t.dim<1> == 4);
		static_assert(t.dim<2> == 5);
		
		// operators
		operatorScalarTest(t);

		//TODO tensor mul
	}


	// rank-3 tensor with intermixed non-vec types:
	// vector-of-symmetric
	{
		//this is a T_ijk = T_ikj, i spans 3 dims, j and k span 2 dims
		using T2S3 = Tensor::_tensori<float, Tensor::index_vec<2>, Tensor::index_sym<3>>;
		
		// list ctor
		T2S3 a = {
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

		auto t = T2S3([](int i, int j, int k) -> float { return 4*i - j*j - k*k; });
		auto verifyAccess = []<typename T>(T & t){
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					for (int k = 0; k < T::template dim<2>; ++k) {
						float e = 4*i - j*j - k*k;
					
						//()()() and any possible merged ()'s
						TEST_EQ(t(i)(j)(k), e);
						TEST_EQ(t(i)(j,k), e);
						TEST_EQ(t(i,j)(k), e);
						TEST_EQ(t(i,j,k), e);
						//[]()() ...
						TEST_EQ(t[i](j)(k), e);
						TEST_EQ(t[i](j,k), e);
						//()[]()
						TEST_EQ(t(i)[j](k), e);
						//()()[]
						TEST_EQ(t(i)(j)[k], e);
						TEST_EQ(t(i,j)[k], e);
						// [][]() []()[] ()[][] [][][]
						TEST_EQ(t[i][j](k), e);
						TEST_EQ(t[i](j)[k], e);
						TEST_EQ(t(i)[j][k], e);
						TEST_EQ(t[i][j][k], e);
					}
				}
			}
		};
		verifyAccess.template operator()<decltype(t)>(t);
		verifyAccess.template operator()<decltype(t) const>(t);

		// operators
		operatorScalarTest(t);
		operatorMatrixTest<T2S3>();
	}
	{
		using T3S3 = Tensor::_tensori<float, Tensor::index_vec<3>, Tensor::index_sym<3>>;
		auto t = T3S3([](int i, int j, int k) -> float { return i+j+k; });
		auto verifyAccess = []<typename T>(T & t){
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					for (int k = 0; k < T::template dim<2>; ++k) {
						float e=i+j+k;
					
						//()()() and any possible merged ()'s
						TEST_EQ(t(i)(j)(k), e);
						TEST_EQ(t(i)(j,k), e);
						TEST_EQ(t(i,j)(k), e);
						TEST_EQ(t(i,j,k), e);
						//[]()() ...
						TEST_EQ(t[i](j)(k), e);
						TEST_EQ(t[i](j,k), e);
						//()[]()
						TEST_EQ(t(i)[j](k), e);
						//()()[]
						TEST_EQ(t(i)(j)[k], e);
						TEST_EQ(t(i,j)[k], e);
						// [][]() []()[] ()[][] [][][]
						TEST_EQ(t[i][j](k), e);
						TEST_EQ(t[i](j)[k], e);
						TEST_EQ(t(i)[j][k], e);
						TEST_EQ(t[i][j][k], e);
					}
				}
			}
		};
		verifyAccess.template operator()<decltype(t)>(t);
		verifyAccess.template operator()<decltype(t) const>(t);
	}
	
	// symmetric-of-vector
	{
		using TS33 = Tensor::_tensori<float, Tensor::index_sym<3>, Tensor::index_vec<3>>;
		auto t = TS33([](int i, int j, int k) -> float { return i+j+k; });
		auto verifyAccess = []<typename T>(T & t){
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					for (int k = 0; k < T::template dim<2>; ++k) {
						float e=i+j+k;
						//()()()
						TEST_EQ(t(i)(j)(k), e);
						TEST_EQ(t(i)(j,k), e);
						TEST_EQ(t(i,j)(k), e);
						TEST_EQ(t(i,j,k), e);
						//[]()() ...
						TEST_EQ(t[i](j)(k), e);
						TEST_EQ(t[i](j,k), e);
						//()[]()
						TEST_EQ(t(i)[j](k), e);
						//()()[]
						TEST_EQ(t(i)(j)[k], e);
						TEST_EQ(t(i,j)[k], e);
						// [][]() []()[] ()[][] [][][]
						TEST_EQ(t[i][j](k), e);
						TEST_EQ(t[i](j)[k], e);
						TEST_EQ(t(i)[j][k], e);
						TEST_EQ(t[i][j][k], e);
					}
				}
			}
		};
		verifyAccess.template operator()<decltype(t)>(t);
		verifyAccess.template operator()<decltype(t) const>(t);
	}

	// symmetric-of-symmetric
	{
		using TS3S3 = Tensor::_tensori<float, Tensor::index_sym<3>, Tensor::index_sym<3>>;
		auto f = [](int i, int j, int k, int l) -> float { return i+j+k+l; };
		auto t = TS3S3(f);
		auto verifyAccess = []<typename T, typename F>(T & t, F f){
			for (int i = 0; i < T::template dim<0>; ++i) {
				for (int j = 0; j < T::template dim<1>; ++j) {
					for (int k = 0; k < T::template dim<2>; ++k) {
						for (int l = 0; l < T::template dim<3>; ++l) {
							float x = f(i,j,k,l);
							TEST_EQ(t(i)(j)(k)(l), x);
							TEST_EQ(t(i,j)(k)(l), x);
							TEST_EQ(t(i,j)(k,l), x);
							TEST_EQ(t(i)(j,k)(l), x);
							TEST_EQ(t(i)(j)(k,l), x);
							TEST_EQ(t(i)(j,k,l), x);
							//TEST_EQ(t(i,j,k)(l), x);
							TEST_EQ(t(i,j,k,l), x);

							TEST_EQ(t[i](j)(k)(l), x);
							TEST_EQ(t[i](j,k)(l), x);
							TEST_EQ(t[i](j)(k,l), x);
							TEST_EQ(t[i](j,k,l), x);
							
							TEST_EQ(t(i)[j](k)(l), x);
							TEST_EQ(t(i)[j](k,l), x);
							
							TEST_EQ(t(i)(j)[k](l), x);
							TEST_EQ(t(i,j)[k](l), x);
							
							TEST_EQ(t(i)(j)(k)[l], x);
							TEST_EQ(t(i,j)(k)[l], x);
							TEST_EQ(t(i)(j,k)[l], x);
							TEST_EQ(t(i,j)(k)[l], x);
							
							TEST_EQ(t[i][j](k)(l), x);
							TEST_EQ(t[i][j](k,l), x);
							
							TEST_EQ(t[i](j)[k](l), x);
							
							TEST_EQ(t[i](j)(k)[l], x);
							TEST_EQ(t[i](j,k)[l], x);
							
							TEST_EQ(t(i)[j][k](l), x);
							
							TEST_EQ(t(i)[j](k)[l], x);
							
							TEST_EQ(t(i)(j)[k][l], x);
							TEST_EQ(t(i,j)[k][l], x);
							
							TEST_EQ(t[i][j][k](l), x);
							TEST_EQ(t[i][j](k)[l], x);
							TEST_EQ(t[i](j)[k][l], x);
							TEST_EQ(t(i)[j][k][l], x);
							
							TEST_EQ(t[i][j][k][l], x);
						}
					}
				}
			}
		};
		verifyAccess.template operator()<decltype(t)>(t, f);
		verifyAccess.template operator()<decltype(t) const>(t, f);
	}

	// TODO antisymmetric of vector
	{
	}

	//antisymmetric of antisymmetric
	{
		using Real = double;
		using Riemann2 = Tensor::_tensori<Real, Tensor::index_asym<2>, Tensor::index_asym<2>>;
		//using Riemann2 = Tensor::_asym<Tensor::_asym<Real, 2>, 2>;	// R_[ij][kl]
		//using Riemann2 = Tensor::_sym<Tensor::_asym<Real, 2>, 2>;	// ... R_(ij)[kl] ... 
		// how would I define R_( [ij] [kl ) ... i.e. R_ijkl = R_klij and R_ijkl = -R_jikl ?
		auto r = Riemann2{{1}};
		static_assert(Riemann2::rank == 4);
		static_assert(Riemann2::dim<0> == 2);
		static_assert(Riemann2::dim<1> == 2);
		static_assert(Riemann2::dim<2> == 2);
		static_assert(Riemann2::dim<3> == 2);
		static_assert(Riemann2::numNestings == 2);
		static_assert(Riemann2::count<0> == 1);
		static_assert(Riemann2::count<1> == 1);
		static_assert(sizeof(Riemann2) == sizeof(Real));
		auto r00 = r(0,0);	// this type will be a ZERO AntiSymRef wrapper around ... nothing ... 
		ECHO(r00);
		TEST_EQ(r00, (Tensor::AntiSymRef<Tensor::_asym<Real, 2>>()));	// r(0,0) is this type
		TEST_EQ(r00, (Tensor::_asym<Real, 2>{}));	// ... and r(0,0)'s operator== accepts its wrapped type
		TEST_EQ(r00(0,0), (Tensor::AntiSymRef<Real>()));	// r(0,0)(0,0) is this
		TEST_EQ(r00(0,0).how, Tensor::AntiSymRefHow::ZERO);
		TEST_EQ(r00(0,0), 0.);
		TEST_EQ(r00(0,1), 0.);
		TEST_EQ(r00(1,0), 0.);
		TEST_EQ(r00(1,1), 0.);
		auto r01 = r(0,1);	// this will point to the positive r.x_y element
		TEST_EQ(r01, (Tensor::_asym<Real, 2>{1}));
		TEST_EQ(r01(0,0), 0);	//why would this get a bad ref?
		TEST_EQ(r01(0,1), 1);
		TEST_EQ(r01(1,0), -1);
		TEST_EQ(r01(1,1), 0);
		auto r10 = r(1,0);
		TEST_EQ(r10, (Tensor::_asym<Real, 2>{-1}));
		TEST_EQ(r10(0,0), 0);
		TEST_EQ(r10(0,1), -1);
		TEST_EQ(r10(1,0), 1);
		TEST_EQ(r10(1,1), 0);
		auto r11 = r(1,1);
		TEST_EQ(r11(0,0), 0.);
		TEST_EQ(r11(0,1), 0.);
		TEST_EQ(r11(1,0), 0.);
		TEST_EQ(r11(1,1), 0.);
	}
	{
		constexpr int N = 3;
		using Riemann3 = Tensor::_tensori<double, Tensor::index_asym<N>, Tensor::index_asym<N>>;
		auto r = Riemann3();
		static_assert(Riemann3::rank == 4);
		static_assert(Riemann3::dim<0> == N);
		static_assert(Riemann3::dim<1> == N);
		static_assert(Riemann3::dim<2> == N);
		static_assert(Riemann3::dim<3> == N);
		static_assert(Riemann3::numNestings == 2);
		static_assert(Riemann3::count<0> == 3);	//3x3 antisymmetric has 3 unique components
		static_assert(Riemann3::count<1> == 3);
		//TODO some future work: R_ijkl = R_klij, so it's also symmetri between 1&2 and 3&4 ...
		// ... and optimizing for those should put us at only 6 unique values instead of 9
		static_assert(sizeof(Riemann3) == sizeof(double) * 9);

		double e = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < i; ++j) {
				for (int k = 0; k < N; ++k) {
					for (int l = 0; l < N; ++l) {
						r(i,j)(k,l) = ++e;
						if (i == j || k == l) {
							TEST_EQ(r(i,j)(k,l), 0);
						} else {
							TEST_EQ(r(i,j)(k,l), e);
							TEST_EQ(r(i,j)(l,k), -e);
							TEST_EQ(r(j,i)(k,l), -e);
							TEST_EQ(r(j,i)(l,k), e);
							
							TEST_EQ(r(i,j,k,l), e);
							TEST_EQ(r(i,j,l,k), -e);
							TEST_EQ(r(j,i,k,l), -e);
							TEST_EQ(r(j,i,l,k), e);
						}
					}
				}
			}
		}

// TODO change tensor generic ctor to (requires tensors  and) accept any type, iterate over write elements, assign one by one.

//		auto m = Tensor::ExpandAllIndexes<Riemann3>([&](Tensor::int4 i) -> double {
//			return r(i);
//		});
	}

	{
		// TODO verify 3- nestings deep of antisym works
	}

	// old libraries' tests
	{
		using Real = double;
		using Vector = Tensor::_tensor<Real,3>;
		
		Vector v = {1,2,3};
		TEST_EQ(v, Tensor::double3(1,2,3));
		
		using Metric = Tensor::_tensori<Real,Tensor::index_sym<3>>;
		Metric g;
		for (int i = 0; i < 3; ++i) {
			g(i,i) = 1;
		}
		TEST_EQ(g, Metric(1,0,1,0,0,1));

		using Matrix = Tensor::_tensor<Real,3,3>;
		Matrix h;
		int index = 0;
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				h(i,j) = ++index;
			}
		}
		TEST_EQ(h, (Matrix{{1,2,3},{4,5,6},{7,8,9}}));

		//iterator access
		int j = 0;
		Tensor::_tensor<Real, 3,3,3> ta;
		for (auto i = ta.begin(); i != ta.end(); ++i) {
			*i = j++;
		}
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				for (int k = 0; k < 3; ++k) {
					if constexpr (std::is_same_v<Tensor::int2::ReadInc<0>, Tensor::int2::ReadIncOuter<0>>) {
						TEST_EQ(ta(i,j,k), k + 3 * (j + 3 * i));
					} else if constexpr (std::is_same_v<Tensor::int2::ReadInc<0>, Tensor::int2::ReadIncInner<0>>) {
						TEST_EQ(ta(i,j,k), i + 3 * (j + 3 * k));
					}
				}
			}
		}

		//subtensor access not working
		Tensor::_tensor<Real,3,3> tb;
		for (auto i = tb.begin(); i != tb.end(); ++i) *i = 2.f;
		TEST_EQ(tb, Matrix(2.f));
		ta(0) = tb;
		TEST_EQ(ta, (Tensor::_tensor<Real,3,3,3>{
			{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}},
			ta(1),//{{1, 10, 19}, {4, 13, 22}, {7, 16, 25}}, // these are whatever the original ta was
			ta(2),//{{2, 11, 20}, {5, 14, 23}, {8, 17, 26}}
		} ));
		Tensor::_tensor<Real, 3> tc;
		for (auto i = tc.begin(); i != tc.end(); ++i) *i = 3.;
		TEST_EQ(Tensor::double3(3), tc);
		ta(0,0) = tc;
		TEST_EQ(ta, (Tensor::_tensor<Real,3,3,3>{
			{{3, 3, 3}, {2, 2, 2}, {2, 2, 2}},
			ta(1),//{{1, 10, 19}, {4, 13, 22}, {7, 16, 25}},
			ta(2),//{{2, 11, 20}, {5, 14, 23}, {8, 17, 26}}
		}));

		//inverse
		Matrix m;
		for (int i = 0; i < m.dim<0>; ++i) {
			for (int j = 0; j < m.dim<1>; ++j) {
				m(i,j) = i == j ? 1 : 0;
			}
		}

		// convert the _sym diagonal to _mat
		// TODO operator== between matrices
		auto d = diagonal(Tensor::_vec<Real, m.dim<0>>(1));
		TEST_EQ(m, d);
		TEST_EQ(m, (Matrix{{1,0,0},{0,1,0},{0,0,1}}));
		TEST_EQ(Tensor::determinant(m), 1);
	}

// does this work?
{
	using namespace Tensor;
	float2 a = {1,2};
	float3 b = a;
	TEST_EQ(b, float3(1,2,0));
}

	// can you vector non-numeric types?
	{
		using namespace Tensor;
		using namespace std;
		// "attempt to use a deleted function"
		// "destructor of '_vec<..., 3>' is implicitly deleted because variant field '' has a non-trivial destructor"
		//auto t = _vec<function<void()>, 3>();
		//auto t = _vec<optional<function<void()>>, 3>();
		//ECHO(t);
	}
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
