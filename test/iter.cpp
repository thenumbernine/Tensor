#include "TensorMath/Tensor.h"

#include "defs.h"

#include <algorithm>

int main() {

	typedef double Real;

	Tensor<Real, Upper<3>> a(1);
	Tensor<Real, Lower<3>> b(2);

	ECHO(a);
	ECHO(b);
	
	std::for_each(a.write.begin(), a.write.end(), [&](IVector i){
		ECHO(i);
		a(i) = b(i);
	});

	ECHO(a);
}

