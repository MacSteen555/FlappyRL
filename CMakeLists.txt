cmake_minimum_required(VERSION 3.15)

# Project name and version
project(FlappyRL VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Build type configurations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Release configuration: -O3 optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug configuration: AddressSanitizer and UndefinedBehaviorSanitizer
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address,undefined -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address,undefined -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /WX- /MP)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Libraries
# ============================================================================

# Core library (foundational utilities)
add_library(core STATIC
    src/core/core.cpp
)
target_include_directories(core PUBLIC include/core)

# Environment Flappy library
add_library(env_flappy STATIC
    src/env_flappy/env_flappy.cpp
)
target_include_directories(env_flappy PUBLIC include/env_flappy)
target_link_libraries(env_flappy PUBLIC core)

# DQN Reinforcement Learning library
add_library(rl_dqn STATIC
    src/rl_dqn/rl_dqn.cpp
)
target_include_directories(rl_dqn PUBLIC include/rl_dqn)
target_link_libraries(rl_dqn PUBLIC core)

# SDL Rendering library
add_library(render_sdl STATIC
    src/render_sdl/render_sdl.cpp
)
target_include_directories(render_sdl PUBLIC include/render_sdl)
target_link_libraries(render_sdl PUBLIC core)

# ============================================================================
# Executables
# ============================================================================

# Training application
add_executable(app_train
    src/app_train/main.cpp
)
target_link_libraries(app_train PRIVATE env_flappy rl_dqn core)

# Play application
add_executable(app_play
    src/app_play/main.cpp
)
target_link_libraries(app_play PRIVATE env_flappy render_sdl core)

# ============================================================================
# Unit Testing (Catch2)
# ============================================================================

option(BUILD_TESTING ON)
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.0
    )
    FetchContent_MakeAvailable(Catch2)
    
    # Enable Catch2 for testing
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
    
    # Example test file structure
    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
    if(TEST_SOURCES)
        catch_discover_tests(${PROJECT_NAME}_tests)
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS core env_flappy rl_dqn render_sdl app_train app_play
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
