#ifndef PARAMETRIZATION_TESTS_TEST_MESHES_H
#define PARAMETRIZATION_TESTS_TEST_MESHES_H

#include <vector>
#include <string>

//A collection of test meshes with certain properties

//Meshes with disk topology
inline std::vector<std::string> disk_topology_meshes() {
    const std::vector<std::string> vec = {
        DATA_DIRECTORY "/camelhead_small.obj",
        DATA_DIRECTORY "/camelhead.obj",
        DATA_DIRECTORY "/nefertiti_small.obj"
    };
    return vec;
}

#endif
