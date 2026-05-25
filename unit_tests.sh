mkdir -p build_unit_tests;
cd build_unit_tests;
cmake ..;
make;
./unit_tests/unit_tests;
