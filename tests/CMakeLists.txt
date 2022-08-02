cmake_minimum_required(VERSION 3.10)
project(parametrization_tests CXX)
set(CMAKE_CXX_STANDARD 14)

# the actual source directory, one up from here
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# path to cmake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/external/cmake")

# set the build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Set build type to Release." FORCE)
endif()

# offer option to enable / disable custom assert
option(FORCE_PARAMETRIZATION_ASSERTS
"This option will enable asserts in the parametrization library, even if NDEBUG is defined and normal asserts are turned off."
ON)
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
find_package(SuiteSparse)
if(SuiteSparse_FOUND)
    add_definitions(-DSUITESPARSE_AVAILABLE)
    include_directories("${SuiteSparse_INCLUDE_DIRS}")
    set(LINKED_LIBS ${LINKED_LIBS} ${SuiteSparse_LIBRARIES})
    set(LINK_TO_LIBRARY ${LINK_TO_LIBRARY} ${SuiteSparse_LIBRARIES})
endif()

# If SuiteSparse is available, add BFF
if(SuiteSparse_FOUND)
    set(BFF_DIR "${PROJECT_SOURCE_DIR}/external/boundary-first-flattening")
    include(BoundaryFirstFlattening)
    set(LINKED_LIBS ${LINKED_LIBS} ${BFF_LIBRARIES})
    set(LINK_TO_LIBRARY ${LINK_TO_LIBRARY} ${BFF_LIBRARIES})
endif()

# Fix a libigl CMake warning
file(REMOVE "${PROJECT_SOURCE_DIR}/external/libigl/cmake/DownloadProject.CMakeLists.cmake.in")
file(COPY "${PROJECT_SOURCE_DIR}/external/cmake/DownloadProject.CMakeLists.cmake.in" DESTINATION "${PROJECT_SOURCE_DIR}/external/libigl/cmake/")

# libigl
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/external/libigl/cmake")
option(LIBIGL_USE_STATIC_LIBRARY     "static libigl"      OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
include(libigl)
set(LINKED_LIBS ${LINKED_LIBS} igl::core igl::opengl_glfw)
set(LINK_TO_LIBRARY ${LINK_TO_LIBRARY} igl::core igl::opengl_glfw)

# make LTO work, but not for debug builds
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_Supported)
if(${IPO_Supported})
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
endif()

# project library and include files
file(GLOB PARAMETRIZATION_SOURCES "${PROJECT_SOURCE_DIR}/include/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/*/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/*/*/*.cpp")
add_library(parametrization STATIC "${PARAMETRIZATION_SOURCES}")
target_include_directories(parametrization INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(parametrization ${LINK_TO_LIBRARY})
set(LINKED_LIBS ${LINKED_LIBS} parametrization)

# Add catch2 unit test framework
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/external/Catch2/contrib")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/Catch2" catch2)
set(LINKED_LIBS ${LINKED_LIBS} Catch2::Catch2)

# Define the data directory for test meshes
add_definitions(-DDATA_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/data")

# Compile the unit test executable
add_executable(parametrization_tests main.cpp)
target_link_libraries(parametrization_tests PUBLIC ${LINKED_LIBS})
target_include_directories(parametrization_tests PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Compile the individual unit tests
file(GLOB SRC_FILES "./src/*.cpp" "./src/*/*.cpp")
# set(SRC_FILES "./src/energy.cpp")
target_sources(parametrization_tests PRIVATE ${SRC_FILES})

# Register tests with Catch2
include(Catch)
catch_discover_tests(parametrization_tests)

