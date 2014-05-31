# anyone who needs to use TensorMath should include this

TENSORMATH_PATH:=$(dir $(lastword $(MAKEFILE_LIST)))

INCLUDE+=$(TENSORMATH_PATH)include

