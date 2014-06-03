# anyone who needs to use Tensor should include this

TENSOR_PATH:=$(dir $(lastword $(MAKEFILE_LIST)))

INCLUDE+=$(TENSOR_PATH)include

