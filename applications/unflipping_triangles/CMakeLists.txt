project(unflipping_triangles)

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} ${LINKED_LIBS})

#Copy meshes into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:unflipping_triangles>/meshes/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/suzanne.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/turbine.obj" "$<TARGET_FILE_DIR:unflipping_triangles>/meshes/")
target_compile_definitions(${PROJECT_NAME} PRIVATE MESH_DIRECTORY="$<TARGET_FILE_DIR:unflipping_triangles>/meshes/")

