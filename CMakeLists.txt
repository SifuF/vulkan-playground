cmake_minimum_required(VERSION 3.22.1)

project(vulkan_playground VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

set(STB_SRC_DIR ${THIRD_PARTY_DIR}/stb_image)

set(GLM_SRC_DIR ${THIRD_PARTY_DIR}/glm)

set(GLFW_LIB_DIR ${THIRD_PARTY_DIR}/glfw/lib-vc2022)
set(GLFW_SRC_DIR ${THIRD_PARTY_DIR}/glfw/include)

set(VULKAN_LIB_DIR ${THIRD_PARTY_DIR}/vulkan/lib)
set(VULKAN_SRC_DIR ${THIRD_PARTY_DIR}/vulkan/include)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${VULKAN_SRC_DIR}
        ${STB_SRC_DIR}
        ${GLFW_SRC_DIR}
        ${GLM_SRC_DIR}
)

set(SOURCES
        src/main.cpp
        src/vulkan_playground.cpp
)

set(HEADERS
        src/vulkan_playground.hpp
)

link_directories(${VULKAN_LIB_DIR})
add_executable(${CMAKE_PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${VULKAN_LIB_DIR}
    ${GLFW_LIB_DIR}
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE 
    vulkan-1.lib
    glfw3.lib
)

file(COPY ${PROJECT_SOURCE_DIR}/img/ DESTINATION ${PROJECT_BINARY_DIR}/img/)
