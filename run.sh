cd build;
cmake -DCMAKE_BUILD_TYPE=Debug ..; # we build in debug mode so that we can use gdb for debugging
make;
./runchecks/test_runner;
