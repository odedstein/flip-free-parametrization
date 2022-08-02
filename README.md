# Parametrization project code

## Compiling C++ code

This project requires CMake 3.10+ to build.
In order to compile the code, do the following steps while in this directory:
* `mkdir build`
* `cd build`
* `cmake ..`
* `make`

After the code is built, you can find the different executables in the subdirectories of
`build/applications/`, for example `build/applications/interactive_deformation`.
To generate all examples present in the submission (except for the interactive ones),
execute `run_all.sh` in this directory (only works on POSIX-like systems with a bash shell).

On Windows, you might need to adjust some of the paths to Windows path conventions.
However, this code _has not been tested on Windows_.
The code is intended to run on macOS (11.2) with clang 12.0.0.
The code has been optimized for macOS (11.2) with an Intel processor, and this is the
platform that was used to generate the results in the paper.

To get the intended performance, you must have the following libraries installed on your
system in standard locations, they must be discoverable with the standard CMake scripts, and
work with your chosen C++ compiler.
There should be fallbacks in the CMake script if the libraries are not available, but they will
alter your performance.
* `SuiteSparse`
* `Intel MKL` or `Apple Accelerate` for `BLAS/LAPACK/LAPACKE`. `OpenBLAS` and `ATLAS` are _not recommended_.
    * If using `Apple Accelerate` (or another BLAS/LAPACK that does not ship with `LAPACKE`), make sure to have `LAPACKE` available and discoverable by `FindLAPACKE`, just `BLAS/LAPACK` is not enough.
* `OpenMP`

For the function `comparison_with_previous` to work, the program needs to be
executed in a bash shell, and GNU timeout must be available.

Some examples rely on input meshes that are not publicly available, or on
datasets that are too large to include.
These meshes are indicated in the `README.txt` file of the respective `meshes/`
directory in the example's `application` folder.

In order to compare to certain previous work, you need to place the binaries for [SLIM](https://github.com/MichaelRabinovich/Scalable-Locally-Injective-Mappings) (called ReweightedARAP), [AKVF](https://github.com/sebastian-claici/AKVFParam) (called AKVFParam), and [ProgressiveParametrization](http://staff.ustc.edu.cn/~fuxm/projects/ProgressivePara/) (called ProgressiveParametrization) into the folder `external/bin`.
Comparisons to these previous methods can not be run if the binaries are not present.
Code for [libigl](https://github.com/libigl/libigl) must be present in `external/libigl/`.
Code for [BFF](https://github.com/GeometryCollective/boundary-first-flattening) must be present in `external/boundary-first-flattening/`.
Code for [Catch2](https://github.com/catchorg/Catch2) must be present in `external/Catch2`.

The unit tests can be found in the subfolder `tests`, and have their own CMake
script that is run similarly.
Go to the subfolder `tests`, and then run:
* `mkdir build`
* `cd build`
* `cmake ..`
* `make`
* `./parametrization_tests`
_Make sure to run the unit tests and verify that the code works before running 
any experiments!_


## License

The license for the project can be found in LICENSE.txt
Licenses for included software can be found in LICENSE.txt.
Attributions for included assets can be found in the same directory as the asset in a README
or LICENSE file.
