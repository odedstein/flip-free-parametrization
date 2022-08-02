project(uv_maps)

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} ${LINKED_LIBS})

#Copy meshes into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:uv_maps>/meshes/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/bunny.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/camelhead.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/goathead.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/mushroom.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/nefertiti.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/strawberry.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/tree.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/brucewick.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/mountain.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/brain.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/cactus.obj" "$<TARGET_FILE_DIR:uv_maps>/meshes/")
target_compile_definitions(${PROJECT_NAME} PRIVATE MESH_DIRECTORY="$<TARGET_FILE_DIR:uv_maps>/meshes/")

