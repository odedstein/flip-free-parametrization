# Compiling and running unit tests

In order to compile the code, do the following steps while in this directory:
* `mkdir build`
* `cd build`
* `cmake ..`
* `make`

After the code is built, you can find the unit test executable in
`build/parametrization_tests`, for example `./build/applications/main/main.cpp`.
Run it to make sure the code is working as intended.

On Windows, you might need to adjust some of the paths to Windows path conventions.
However, this code _has not been tested on Windows_, and is intended to run on macOS
(11.1 and higher) and Ubuntu Linux (20.04 and higher) with a C++17-capable clang or gcc
compiler.
