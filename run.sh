cd build;
cmake -DBEARML_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug ..; # we build in debug mode so that we can use gdb for debugging
make;
./runchecks/test_runner;
# -DBEARML_USE_CUDA=OFF to disable CUDA support
