#include "Tensor/Tensor.h"
#include "Common/Test.h"
#include <algorithm>

namespace TupleTests {
	using namespace std;
	using namespace Common;
	using namespace Tensor;

	using I = Index<'i'>;
	using J = Index<'j'>;
	I i;
	J j;

	static_assert(is_same_v<
		GatherIndexesImpl<
			decltype(float3()(i))::IndexTuple,
			make_integer_sequence<int, 1>
		>::indexes,
		tuple<I>
	>);

	static_assert(is_same_v<
		GatherIndexesImpl<
			decltype(float3()(i))::IndexTuple,
			make_integer_sequence<int, 1>
		>::type,
		tuple<
			pair<
				I,
				integer_sequence<int, 0>
			>
		>
	>);

	static_assert(is_same_v<
		GatherIndexesImpl<
			decltype(float3x3()(i,j))::IndexTuple,
			make_integer_sequence<int, 2>
		>::indexes,
		tuple<I, J>
	>);
	
	static_assert(is_same_v<
		decltype(float3x3()(i,j))::GatheredIndexes,
		tuple<
			pair<
				I,
				integer_sequence<int, 0>
			>,
			pair<
				J,
				integer_sequence<int, 1>
			>
		>
	>);

	namespace test1 {
		using IndexTuple = std::tuple<I>;
		using GatheredIndexes = Tensor::GatherIndexes<IndexTuple>;
		using GetAssignVsSumGatheredLocs = Common::tuple_get_filtered_indexes_t<GatheredIndexes, Tensor::HasMoreThanOneIndex>;
		using SumIndexSeq = Tensor::GetIndexLocsFromGatherResult<typename GetAssignVsSumGatheredLocs::has, GatheredIndexes>;
		using AssignIndexSeq = Tensor::GetIndexLocsFromGatherResult<typename GetAssignVsSumGatheredLocs::hasnot, GatheredIndexes>;
		static_assert(SumIndexSeq::size() == 0);
		static_assert(std::is_same_v<AssignIndexSeq, std::integer_sequence<int, 0>>);
	}
	namespace test2 {
		using IndexTuple = std::tuple<I,J>;
		using GatheredIndexes = Tensor::GatherIndexes<IndexTuple>;
		using GetAssignVsSumGatheredLocs = Common::tuple_get_filtered_indexes_t<GatheredIndexes, Tensor::HasMoreThanOneIndex>;
		using SumIndexSeq = Tensor::GetIndexLocsFromGatherResult<typename GetAssignVsSumGatheredLocs::has, GatheredIndexes>;
		using AssignIndexSeq = Tensor::GetIndexLocsFromGatherResult<typename GetAssignVsSumGatheredLocs::hasnot, GatheredIndexes>;
		static_assert(SumIndexSeq::size() == 0);
		static_assert(std::is_same_v<AssignIndexSeq, std::integer_sequence<int, 0, 1>>);
	}
	namespace test3 {
		using IndexTuple = std::tuple<I,I>;
		static_assert(Common::tuple_find_v<I, std::tuple<>> == -1);
		static_assert(Common::tuple_find_v<I, std::tuple<I>> == 0);
		static_assert(Common::tuple_find_v<I, std::tuple<I,I>> == 0);
		// GatheredIndexes == GatheredIndexesImpl::type so ...
		static_assert(std::is_same_v<
			typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::Next::Next::type,
			std::tuple<>
		>);
		static_assert(std::is_same_v<
			typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::Next::Next::indexes,
			std::tuple<>
		>);	
		static_assert(-1 == Common::tuple_find_v<I, typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::Next::Next::indexes>);	
		static_assert(std::is_same_v<
			typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::Next::type,
			std::tuple<
				std::pair<
					I,
					std::integer_sequence<int, 1>
				>
			>	
		>);
		static_assert(std::is_same_v<
			typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::type,
			std::tuple<
				std::pair<
					I,
					std::integer_sequence<int, 0, 1>
				>
			>	
		>);	
		static_assert(std::is_same_v<typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::indexes, std::tuple<I>>);
		static_assert(std::is_same_v<typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::Next::indexes, std::tuple<I>>);
		static_assert(std::is_same_v<typename Tensor::GatherIndexesImpl<IndexTuple, std::make_integer_sequence<int, 2>>::Next::Next::indexes, std::tuple<>>);
		using GatheredIndexes = Tensor::GatherIndexes<IndexTuple>;
		static_assert(std::is_same_v<
			GatheredIndexes,
			std::tuple<
				std::pair<
					I,
					std::integer_sequence<int, 0, 1>
				>
			>
		>);
		using GetAssignVsSumGatheredLocs = Common::tuple_get_filtered_indexes_t<
			GatheredIndexes,
			Tensor::HasMoreThanOneIndex
		>;
		using SumIndexSeq = Tensor::GetIndexLocsFromGatherResult<typename GetAssignVsSumGatheredLocs::has, GatheredIndexes>;
		using AssignIndexSeq = Tensor::GetIndexLocsFromGatherResult<typename GetAssignVsSumGatheredLocs::hasnot, GatheredIndexes>;
		static_assert(AssignIndexSeq::size() == 0);
		static_assert(std::is_same_v<
			SumIndexSeq,
			std::integer_sequence<int, 0, 1>
		>);
	}
}

void test_Index() {
	//index assignment
	{
		auto a = Tensor::double3(1);
		auto b = Tensor::double3(2);

		TEST_EQ(a.rank, 1);
		TEST_EQ(b.rank, 1);

		TEST_EQ(a, (Tensor::double3(1)));
		TEST_EQ(b, (Tensor::double3(2)));

		Tensor::Index<'i'> i;
		a(i) = b(i);

		TEST_EQ(a, (Tensor::double3(2)));
	}
	
	{
		//make sure 2D swizzling works
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 m;
		m(1,0) = 1;
		ECHO(m);
		m(i,j) = m(j,i);
		ECHO(m);
		TEST_EQ(m(0, 1), 1);
	}

	{
		//make sure 3D swizzling works
		//this verifies the mapping between indexes in tensor assignment (since the 2D case is always a cycle of at most period 2, i.e. its own inverse)
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::Index<'k'> k;
		Tensor::_tensor<double, 3, 3, 3> s;
		s(0,1,0) = 1;
		ECHO(s);
		s(i,j,k) = s(j,k,i);	//s(0,0,1) = s(0,1,0)
		TEST_EQ(s(0,0,1), 1);
		ECHO(s);
	}

	{
		//arithemetic operations
		Tensor::Index<'i'> i;
		
		Tensor::double3 a = {1,2,3};
		Tensor::double3 b = {5,7,11};

		Tensor::double3 c;
		c(i) = a(i) + b(i);
		TEST_EQ(c, (Tensor::double3(6,9,14)));
	}

	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 a = {{1,2,3},{4,5,6},{7,8,9}};

		// transpose
		a(i,j) = a(j,i);
		TEST_EQ(a, (Tensor::double3x3{{1,4,7},{2,5,8},{3,6,9}}));

		// add to transpose and self-assign
		a(i,j) = a(i,j) + a(j,i);
		TEST_EQ(a, (Tensor::double3x3{{2,6,10},{6,10,14},{10,14,18}}));
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 a = {{1,2,3},{4,5,6},{7,8,9}};

		// symmetrize using index notation
		Tensor::double3x3 b;
		b(i,j) = .5 * (a(i,j) + a(j,i));
		TEST_EQ(b, makeSym(a));
		// explicitly-specified storage
		auto c = (.5 * (a(i,j) - a(j,i))).assignR<Tensor::double3a3>(i,j);
		static_assert(std::is_same_v<Tensor::double3a3, decltype(c)>);
		TEST_EQ(c, makeAsym(a));
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::Index<'k'> k;
		//assignR
		// TODO put compile-fail tests in their own cpp file and have a script assert the compiler fails
#if 0 	// ASSERT_FAILURE  ranks of 'a' and index call-operator don't match so static-assert failure 
		{
			Tensor::float3x3 a;
			auto c = a(i);
		}
#endif
#if 0	// ASSERT_FAILURE  compile fail because a(j,i) rank doesn't match assign(i,j,k) rank 
		{
			Tensor::float3x3 a;
			Tensor::_tensor<float,3,3,3> d;
			d = a(j,i).assign(i,j,k);
		}
#endif	
#if 0	// ASSERT_FAILURE  compile fail because assign(i,j) rank doesn't match d(i,j,k) rank 
		{
			Tensor::float3x3 a;
			Tensor::_tensor<float,3,3,3> d;
			d = a(j,i).assign(i,j);
		}
#endif	
		{
			Tensor::float3x3 a;
			auto c = a(i,j).assignR<Tensor::float3x3>(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x3>);
		}
		{
			Tensor::float3x3 a;
			auto c = a(i,j).assignR<Tensor::float3x3>(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x3>);
		}	
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assignR<Tensor::float2x3>(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assignR<Tensor::float3x2>(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x2>);
		}
#if 0	// ASSERT_FAILURE  dims don't match, should compile-fail
		//  mind you under clang this does make a compile-fail, but doesn't point to this line, even tho if0'ing it out makes compile succeed.
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assignR<Tensor::float2x3>(j,i);
			ECHO(c);
		}
#endif
#if 0	// ASSERT_FAILURE  dims don't match, should compile-fail
		{
			Tensor::float2x3 a;
			Tensor::float3x2 c;
			c(i,j) = a(i,j);
			ECHO(c);
		}
#endif
#if 0	// ASSERT_FAILURE  dims don't match, should compile-fail
		{
			Tensor::float2x3 a;
			Tensor::float2x3 c;
			c(i,j) = a(j,i);
			ECHO(c);
		}
#endif	
#if 0	// ASSERT_FAILURE dims don't match, so static-asser failure 
		{
			Tensor::float2x3 a;
			Tensor::float3x3 c;
			c(i,j) = a(j,i);	
		}
#endif
		//assignI
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assignI();
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		{
			Tensor::float2x3 a;
			auto c = a(j,i).assignI();
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		//assign
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assign(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		{
			Tensor::float2x3 a;
			auto c = a(i,j).assign(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x2>);
		}
		{
			Tensor::float2x3 a;
			auto c = (2.f * a(i,j)).assign(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		{
			Tensor::float2x3 a;
			auto c = (2.f * a(i,j)).assign(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x2>);
		}	
		{
			Tensor::float2x3 a;
			auto c = (a(i,j) * 2.f).assign(i,j);
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		{
			Tensor::float2x3 a;
			auto c = (a(i,j) * 2.f).assign(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float3x2>);
		}		
		{
			Tensor::float2x3 a;
			Tensor::float3x2 b;
			auto c = (a(j,i) + b(i,j)).assign(j,i);
			static_assert(std::is_same_v<decltype(c), Tensor::float2x3>);
		}
		// make sure inter-index permutations work
		// since right now tensor+tensor operator just uses the lhs
		{
			Tensor::_tensor<float,2,3,4> a;

/* d_ijk = a_ijk */static_assert(std::is_same_v<decltype(a(i,j,k).assign(i,j,k)), Tensor::_tensor<float,2,3,4>>);
/* d_ikj = a_ijk */static_assert(std::is_same_v<decltype(a(i,j,k).assign(i,k,j)), Tensor::_tensor<float,2,4,3>>);
/* d_jik = a_ijk */static_assert(std::is_same_v<decltype(a(i,j,k).assign(j,i,k)), Tensor::_tensor<float,3,2,4>>);
/* d_jki = a_ijk */static_assert(std::is_same_v<decltype(a(i,j,k).assign(j,k,i)), Tensor::_tensor<float,3,4,2>>);
/* d_kij = a_ijk */static_assert(std::is_same_v<decltype(a(i,j,k).assign(k,i,j)), Tensor::_tensor<float,4,2,3>>);
/* d_kji = a_ijk */static_assert(std::is_same_v<decltype(a(i,j,k).assign(k,j,i)), Tensor::_tensor<float,4,3,2>>);

			Tensor::_tensor<float,4,2,3> b;
			// d_ijk = a_ijk + b_kij
			// so d's dims are a's dims ...
			//  and works only if 
			//   b's 1st dim matches a's 3rd dim
			//   b's 2nd dim matches a's 1st dim
			//   b's 3nd dim matches a's 2st dim
			auto ab1 = (a(i,j,k) + b(k,i,j)).assign(i,j,k);
			static_assert(std::is_same_v<decltype(ab1), Tensor::_tensor<float,2,3,4>>);
			
			auto ab2 = (a(i,j,k) + b(k,i,j)).assign(k,i,j);
			static_assert(std::is_same_v<decltype(ab2), Tensor::_tensor<float,4,2,3>>);

			Tensor::_tensor<float,4,3,2> c;
			auto abc = (a(i,j,k) + b(k,i,j) + c(k,j,i)).assign(j,i,k);
			static_assert(std::is_same_v<decltype(abc), Tensor::_tensor<float,3,2,4>>);
		}
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		Tensor::double3x3 a = {{1,2,3},{4,5,6},{7,8,9}};

		// symmetrize using index notation
		Tensor::double3x3 b;
		b(i,j) = .5 * (a(i,j) + a(j,i));
		TEST_EQ(b, makeSym(a));
		// implicit storage type, for now picks the worst case
		auto c = (.5 * (a(i,j) - a(j,i))).assign(i,j);
		static_assert(std::is_same_v<Tensor::double3x3, decltype(c)>);
		TEST_EQ(c, makeAsym(a));
	}
	// trace of tensors ... doesn't use references but instead uses cached intermediate tensors stored in the expression-tree
	{
		Tensor::Index<'i'> i;
		auto a = Tensor::float3x3([](int i, int j) -> float { return 1 + j + 3 * i; });
		// zero indexes == scalar result of a trace
		//  Should IndexAccess need to wrap a fully-traced object?  or should it immediately become a Scalar?
		//  I think the latter cuz why wait for .assign()?
		auto tra = a(i,i);
		static_assert(std::is_same_v<decltype(tra), float>);
		TEST_EQ(tra, 15);
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		auto a = Tensor::_tensorr<float, 3, 3>();
		auto b = a(i,i,j).assignR<Tensor::float3>(j);
		static_assert(std::is_same_v<decltype(b), Tensor::float3>);
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		auto a = Tensor::_tensorr<float, 3, 3>();
		auto b = a(i,i,j).assign(j);
		static_assert(std::is_same_v<decltype(b), Tensor::float3>);
	}
	{
		Tensor::Index<'i'> i;
		Tensor::Index<'j'> j;
		
		Tensor::double3 a = {1,2,3};
		Tensor::double3 b = {5,7,11};

		Tensor::double3x3 c;
		c(i,j) = a(i) * b(j);
		TEST_EQ(c, Tensor::double3x3({{5,7,11},{10,14,22},{15,21,33}}));
	}
	{
		Tensor::Index<'i'> i;
		
		Tensor::double3 a = {1,2,3};
		Tensor::double3 b = {5,7,11};

		// ok, trace needed a fully dif avenue to work with IndexAccess
		//  so will contracting alll indexes in tensor-mul
		auto c = a(i) * b(i);
		TEST_EQ(c, 52);
	}
// TODO DO enforce dimension constraints between expression operations
// and then require Index to specify subrank, or just grab the subset<> of the tensor.
// TODO sub-tensor casting, not just sub-vector.  return tensor-of-refs. 	
#if 0 // TODO
	{
		Tensor::double3x3 a;

		//outer product
		a(i,j) = b(i) * c(j);

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				TEST_EQ(a(i,j), b(i) * c(j));
			}
		}

		//exterior product
		a(i,j) = b(i) * c(j) - b(j) * c(i);

		//inner product?
		real dot = b(i) * c(i);

		//matrix multiplication
		c(i) = a(i,j) * b(j);
	
		//discrete differentiation?
		c(i) = (a(i+1) - a(i-1)) / (2 * dx)
	}
#endif
}

#if 0
int main() {
	test_Index();
}
#endif
