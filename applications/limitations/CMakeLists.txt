project(limitations)

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} ${LINKED_LIBS})

#Copy meshes into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:limitations>/meshes/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/helmet.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/clowncollar.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/clowncollar_embedded.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/clowncollar_cover.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/monkey.mesh" "$<TARGET_FILE_DIR:limitations>/meshes/")
target_compile_definitions(${PROJECT_NAME} PRIVATE MESH_DIRECTORY="$<TARGET_FILE_DIR:limitations>/meshes/")

