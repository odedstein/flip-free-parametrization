project(deformation)

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} ${LINKED_LIBS})

#Copy meshes into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:deformation>/meshes/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/worm.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/worm.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/octopus.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/octopus.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/crocodile.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/crocodile.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/pants.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/pants.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/arm.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/arm.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/octopus-vol.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/armadillo.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/hand.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/stuffedtoy.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/hand.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/slug.mesh" "$<TARGET_FILE_DIR:deformation>/meshes/")
target_compile_definitions(${PROJECT_NAME} PRIVATE MESH_DIRECTORY="$<TARGET_FILE_DIR:deformation>/meshes/")

