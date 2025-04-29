test:
	g++ -O3 ./tests/testing.cpp ./src/ELCClassifier.cpp ./src/TreeNode.cpp -o build/test.out

so:
	python3 ./src/setup.py build
	mv ./build/lib*/dec* ./examples/cart-elc-experiments/decision_tree.so
clean:
	rm -rf ./build/* 2>/dev/null



experiments:
	python3 ./src/setup.py build
	mv ./build/lib*/dec* ./examples/cart-elc-experiments/decision_tree.so
	cd examples/cart-elc-experiments && make
