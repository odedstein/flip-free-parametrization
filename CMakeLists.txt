cmake_minimum_required(VERSION 3.10)
project(parametrization CXX)
set(CMAKE_CXX_STANDARD 14)

# path to cmake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/external/cmake")

# set the build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Set build type to Release." FORCE)
endif()

# offer option to enable / disable custom assert
option(FORCE_PARAMETRIZATION_ASSERTS
"This option will enable asserts in the parametrization library, even if NDEBUG is defined and normal asserts are turned off."
OFF)
if(${FORCE_PARAMETRIZATION_ASSERTS})
        add_definitions(-DPARAMETRIZATIONS_DEBUG)
endif()

# activate native optimization if available
if(NOT APPLE) #Libigl viewer does not work with Apple native.
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-march=native" MARCH_NATIVE_SUPPORTED)
    if(MARCH_NATIVE_SUPPORTED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# everything that should be linked to the parametrization library
set(LINK_TO_LIBRARY "")

# the variable where all linked libraries are written into
set(LINKED_LIBS "")

# BLAS/LAPACK
include(BLAS_LAPACK)
set(LINKED_LIBS ${LINKED_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
set(LINK_TO_LIBRARY ${LINKED_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    
# OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    add_definitions(-DOPENMP_AVAILABLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(LINKED_LIBS ${LINKED_LIBS} ${OpenMP_CXX_LIBRARIES})
    set(LINK_TO_LIBRARY ${LINK_TO_LIBRARY} ${OpenMP_CXX_LIBRARIES})
endif()

# SuiteSparse
if(BLAS_FOUND AND LAPACK_FOUND)
    find_package(SuiteSparse)
    if(SuiteSparse_FOUND)
        add_definitions(-DSUITESPARSE_AVAILABLE)
        include_directories("${SuiteSparse_INCLUDE_DIRS}")
        set(LINKED_LIBS ${LINKED_LIBS} ${SuiteSparse_LIBRARIES})
        set(LINK_TO_LIBRARY ${LINK_TO_LIBRARY} ${SuiteSparse_LIBRARIES})
    endif()
endif()

# If SuiteSparse is available, add BFF
if(SuiteSparse_FOUND)
    set(BFF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/boundary-first-flattening")
    include(BoundaryFirstFlattening)
    set(LINKED_LIBS ${LINKED_LIBS} ${BFF_LIBRARIES})
    set(LINK_TO_LIBRARY ${LINK_TO_LIBRARY} ${BFF_LIBRARIES})
endif()

# Fix a libigl CMake warning
file(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/external/libigl/cmake/DownloadProject.CMakeLists.cmake.in")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/external/cmake/DownloadProject.CMakeLists.cmake.in" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/external/libigl/cmake/")

# libigl
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/external/libigl/cmake")
option(LIBIGL_USE_STATIC_LIBRARY     "static libigl"      OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)
include(libigl)
set(LINKED_LIBS ${LINKED_LIBS} igl::core igl::opengl_glfw igl::png)
set(LINK_TO_LIBRARY ${LINK_TO_LIBRARY} igl::core igl::opengl_glfw igl::png)

# make LTO work, but not for debug builds
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_Supported)
if(${IPO_Supported})
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
endif()

# project library and include files
file(GLOB PARAMETRIZATION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*/*/*.cpp")
add_library(parametrization STATIC "${PARAMETRIZATION_SOURCES}")
target_include_directories(parametrization INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(parametrization ${LINK_TO_LIBRARY})
set(LINKED_LIBS ${LINKED_LIBS} parametrization)

# Check whether SLIM/AKVF/ProgressiveParametrization binaries exist, and pass them as constants to the application if they exist. you must set the EXT_THREAD_NUM variable to the optimal openmp core value for your configuration for optimal results.
set(EXT_CORE_NUM 4)
add_definitions(-DEXT_CORE_NUM="${EXT_CORE_NUM}")
set(SLIM_BINARY "${CMAKE_CURRENT_SOURCE_DIR}/external/bin/ReweightedARAP")
if(EXISTS "${SLIM_BINARY}")
    find_package_message(cSLIM_BINARY "Found SLIM binary: ${SLIM_BINARY}" "${SLIM_BINARY}")
    add_definitions(-DSLIM_BINARY="${SLIM_BINARY}")
else()
    message("Could NOT find SLIM binary.")
endif()
set(AKVF_BINARY "${CMAKE_CURRENT_SOURCE_DIR}/external/bin/AKVFParam")
if(EXISTS "${AKVF_BINARY}")
    find_package_message(cAKVF_BINARY "Found AKVF binary: ${AKVF_BINARY}" "${AKVF_BINARY}")
    add_definitions(-DAKVF_BINARY="${AKVF_BINARY}")
else()
    message("Could NOT find AKVF binary.")
endif()
set(ProgressiveParametrizations_BINARY "${CMAKE_CURRENT_SOURCE_DIR}/external/bin/ProgressiveParametrization")
if(EXISTS "${AKVF_BINARY}")
    find_package_message(cProgressiveParametrizations_BINARY "Found ProgressiveParametrization binary: ${ProgressiveParametrizations_BINARY}" "${ProgressiveParametrizations_BINARY}")
    add_definitions(-DProgressiveParametrizations_BINARY="${ProgressiveParametrizations_BINARY}")
else()
    message("Could NOT find ProgressiveParametrizations binary.")
endif()

# Add the project files.
# If you want to use this as a library, comment this out, include this CMake
# file in your project, and link the libraries parametrization and libigl.
add_subdirectory("applications")
