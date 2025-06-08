file(GLOB_RECURSE TEST_SOURCES "t*.cpp")

# Create a test executable
add_executable(unittests
    ${TEST_SOURCES}
)

# Set output directory to bin
set_target_properties(unittests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy both scripts to the build directory
configure_file(
    ${CMAKE_SOURCE_DIR}/scripts/rename-project.sh
    ${CMAKE_BINARY_DIR}/scripts/rename-project.sh
    COPYONLY
)
configure_file(
    ${CMAKE_SOURCE_DIR}/scripts/rename_project.py
    ${CMAKE_BINARY_DIR}/scripts/rename_project.py
    COPYONLY
)

# Make the scripts executable
execute_process(
    COMMAND chmod +x ${CMAKE_BINARY_DIR}/scripts/rename-project.sh
    COMMAND chmod +x ${CMAKE_BINARY_DIR}/scripts/rename_project.py
)

# Link against Google Test and required libraries
target_link_libraries(unittests
    GTest::gtest_main
    MLIRStandaloneCombined  # Link against our combined library
    stdc++fs  # Required for std::filesystem
)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(unittests) 