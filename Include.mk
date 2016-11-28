TENSOR_PATH:=$(dir $(lastword $(MAKEFILE_LIST)))
INCLUDE+=$(TENSOR_PATH)include
