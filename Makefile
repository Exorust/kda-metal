kda: main.m kda.metal
	clang -O2 -fobjc-arc -Wall main.m -framework Metal -framework Foundation -o kda

test: kda
	./kda

.PHONY: test
