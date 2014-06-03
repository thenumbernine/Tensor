#pragma once

/*
object used as a placeholder when summing across indexes
usage:
Tensor<float, Upper<3>, Lower<3>> a;
Tensor<float, Upper<3>> b;
Tensor<float, Upper<3>> c;
Index i,j;
c(i) = a(i,j) * b(j) <-performs multiplication of matrix a and vector b
*/

struct Index {
};

template<typename Tensor_>
struct IndexAccess {
	typedef Tensor_ Tensor;
	Tensor *tensor;
	Index *index;	//vector ...
	IndexAccess(Tensor *tensor_, Index *index_) 
	: tensor(tensor_)
	{
	}

	IndexAccess(const IndexAccess &read);
	
	template<typename TensorB>
	IndexAccess(const IndexAccess<TensorB> &read);

	template<typename TensorB>
	IndexAccess &operator=(const IndexAccess<TensorB> &read) {
		static_assert(std::is_same<typename Tensor::DerefType, typename TensorB::DerefType>::value, "read and write indexes must match");
		std::for_each(tensor->write().begin(), tensor->write().end(),
		[&](typename Tensor::DerefType i) {
			(*tensor)(i) = (*read.tensor)(i);
		});
		return *this;
	}

};

template<typename Tensor>
template<typename TensorB>
IndexAccess<Tensor>::IndexAccess(const IndexAccess<TensorB> &read) {
	this->operator=(read);
}

template<typename Tensor>
IndexAccess<Tensor>::IndexAccess(const IndexAccess<Tensor> &read) 
: tensor(read.tensor)
, index(read.index)
{
}


