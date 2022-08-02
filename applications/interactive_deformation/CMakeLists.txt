project(interactive_deformation)

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} ${LINKED_LIBS})

#Copy meshes into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:interactive_deformation>/meshes/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/octopus.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/octopus.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/mouth.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/mouth.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/robot.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/robot.png" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/cow.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/fish.mesh" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/hand.mesh" "$<TARGET_FILE_DIR:interactive_deformation>/meshes/")
target_compile_definitions(${PROJECT_NAME} PRIVATE MESH_DIRECTORY="$<TARGET_FILE_DIR:interactive_deformation>/meshes/")

#Copy matcap into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:interactive_deformation>/matcaps/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/matcaps/clay.png" "$<TARGET_FILE_DIR:interactive_deformation>/matcaps/")
target_compile_definitions(${PROJECT_NAME} PRIVATE MATCAPS_DIRECTORY="$<TARGET_FILE_DIR:interactive_deformation>/matcaps/")
