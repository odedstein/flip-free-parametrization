project(volumetric_map)

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} ${LINKED_LIBS})

#Copy meshes into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:volumetric_map>/meshes/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-0.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-1.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-2.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-3.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-4.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-5.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-6.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-7.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-8.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/human-9.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/bear.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/bear-0.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/bear-1.obj" "$<TARGET_FILE_DIR:volumetric_map>/meshes/")
target_compile_definitions(${PROJECT_NAME} PRIVATE MESH_DIRECTORY="$<TARGET_FILE_DIR:volumetric_map>/meshes/")

