project(comparison_with_previous)

file(GLOB SRCFILES *.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} ${LINKED_LIBS})

#Copy meshes into target directory
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:comparison_with_previous>/meshes/")
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/bar_sin.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/camel.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/bread.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/car.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/cat.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/cow.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/elephant.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/falconstatue.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/hand.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/horse.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/snake.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/tooth.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/strawberry.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/tree.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/triceratops.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/brain.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/slime.obj" "${CMAKE_CURRENT_SOURCE_DIR}/meshes/deer.obj" "$<TARGET_FILE_DIR:comparison_with_previous>/meshes/")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/meshes/pegasus.obj")
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/meshes/pegasus.obj" "$<TARGET_FILE_DIR:comparison_with_previous>/meshes/")
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE MESH_DIRECTORY="$<TARGET_FILE_DIR:comparison_with_previous>/meshes/")

