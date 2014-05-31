SOURCES=$(shell find . -name "*.cpp" -type f)
OBJS=$(patsubst %.cpp,%.o,$(SOURCES))
DISTS=$(patsubst %.o,%.out,$(OBJS))

# TODO exceptions are in Common in GLApp .. put them somewhere more general?
CC=clang++
CFLAGS=-c -std=c++11 -Wall -I../include -I../../Common/include -O0 -mfix-and-continue -gdwarf-2
.PHONY: all

LD=clang++
LDFLAGS=

all: $(DISTS)

%.o: %.cpp ../include/TensorMath/*
	$(CC) $(CFLAGS) -o $@ $<

%.out: %.o
	$(LD) $(LDFLAGS) -o $@ $<

clean:
	-rm $(OBJS) $(DISTS)
