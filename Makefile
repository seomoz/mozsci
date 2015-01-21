clean:
	# Remove the build
	rm -rf build dist
	# And all of our pyc files
	rm -f mozsci/*.pyc test/*.pyc
	# All compiled files
	rm -f mozsci/*.so mozsci/spearmanr_by_fast.cpp mozsci/_c_utils.cpp
	# And lastly, .coverage files
	rm -f .coverage

test: nose

nose:
	rm -rf .coverage
	nosetests --exe --cover-package=mozsci --with-coverage --cover-branches -v --cover-erase 

unittest:
	python -m unittest discover -s test

# build inplace for unit tests to pass (since they are run from this
# top level directory we need the .so files to be in the src tree
# when they run.
build: clean
	python setup.py build_ext --inplace

install: build
	python setup.py install
