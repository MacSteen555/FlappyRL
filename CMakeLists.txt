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

# Find SDL2
# First, try manual path for MinGW (if user sets SDL2_ROOT or default location)
set(SDL2_ROOT_PATH "")
if(DEFINED ENV{SDL2_ROOT})
    set(SDL2_ROOT_PATH "$ENV{SDL2_ROOT}")
elseif(EXISTS "C:/dev/SDL2")
    set(SDL2_ROOT_PATH "C:/dev/SDL2")
endif()

if(SDL2_ROOT_PATH AND EXISTS "${SDL2_ROOT_PATH}/x86_64-w64-mingw32/include/SDL2/SDL.h")
    # Include directory should point to the parent of SDL2 subdirectory
    set(SDL2_INCLUDE_DIR "${SDL2_ROOT_PATH}/x86_64-w64-mingw32/include" CACHE PATH "SDL2 include path")
    set(SDL2_LIBRARY "${SDL2_ROOT_PATH}/x86_64-w64-mingw32/lib/libSDL2.dll.a" CACHE FILEPATH "SDL2 library")
    # For console apps, we handle main ourselves (no SDL2main needed)
    set(SDL2_LIBRARIES ${SDL2_LIBRARY})
    set(SDL2_INCLUDE_DIRS ${SDL2_INCLUDE_DIR})
    set(SDL2_FOUND TRUE)
    message(STATUS "Found SDL2 at: ${SDL2_ROOT_PATH}")
    message(STATUS "SDL2 include dir: ${SDL2_INCLUDE_DIR}")
    message(STATUS "SDL2 library: ${SDL2_LIBRARY}")
    message(STATUS "SDL2main library: ${SDL2_MAIN_LIBRARY}")
else()
    find_package(SDL2 QUIET)
    if(NOT SDL2_FOUND)
        # Try to find SDL2 via pkg-config (Linux/macOS)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(SDL2 QUIET sdl2)
        endif()
    endif()
endif()

# If SDL2 not found, provide instructions
if(NOT SDL2_FOUND)
    message(WARNING "SDL2 not found. Install SDL2 to enable visualization.")
    message(STATUS "")
    message(STATUS "For MinGW on Windows:")
    message(STATUS "  1. Download: https://github.com/libsdl-org/SDL/releases")
    message(STATUS "  2. Get: SDL2-devel-2.x.x-mingw.tar.gz")
    message(STATUS "  3. Extract to: C:\\dev\\SDL2 (or any location)")
    message(STATUS "  4. Set environment variable: SDL2_ROOT=C:\\dev\\SDL2")
    message(STATUS "  5. Re-run cmake")
    message(STATUS "")
    message(STATUS "Or use MSYS2: pacman -S mingw-w64-x86_64-SDL2")
else()
    target_link_libraries(render_sdl PUBLIC ${SDL2_LIBRARIES})
    target_include_directories(render_sdl PUBLIC ${SDL2_INCLUDE_DIRS})
    target_compile_definitions(render_sdl PUBLIC HAVE_SDL2 SDL_MAIN_HANDLED)
    message(STATUS "SDL2 found - visualization enabled!")
endif()

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
