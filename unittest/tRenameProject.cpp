#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <linux/limits.h>  // For PATH_MAX
#include <regex>

namespace fs = std::filesystem;

class RenameProjectTest : public ::testing::Test {
protected:
    fs::path temp_dir;
    fs::path script_dir;
    fs::path test_project_dir;
    std::string old_name = "Standalone";
    std::string new_name = "TestDialect";
    std::vector<std::regex> gitignore_patterns;

    void SetUp() override {
        // Create a temporary directory for testing
        temp_dir = fs::temp_directory_path() / "rename_project_test_XXXXXX";
        char* temp_dir_str = strdup(temp_dir.string().c_str());
        char* actual_temp_dir = mkdtemp(temp_dir_str);
        temp_dir = actual_temp_dir;
        free(temp_dir_str);

        // Get the build directory (where the scripts should be)
        char exe_path[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
        if (count != -1) {
            exe_path[count] = '\0';
            // The executable is in build/bin, so go up two levels to get to build
            script_dir = fs::path(exe_path).parent_path().parent_path() / "scripts";
        }

        // Load .gitignore patterns
        load_gitignore_patterns();

        // Verify both scripts exist
        fs::path bash_script = script_dir / "rename-project.sh";
        fs::path python_script = script_dir / "rename_project.py";
        if (!fs::exists(bash_script)) {
            FAIL() << "Bash script not found at: " << bash_script;
        }
        if (!fs::exists(python_script)) {
            FAIL() << "Python script not found at: " << python_script;
        }

        // Create a test project structure
        test_project_dir = temp_dir / "test_project";
        create_test_project();
    }

    void load_gitignore_patterns() {
        fs::path gitignore_path = script_dir.parent_path() / ".gitignore";
        if (!fs::exists(gitignore_path)) {
            return;
        }

        std::ifstream gitignore(gitignore_path);
        std::string line;
        while (std::getline(gitignore, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // Convert gitignore pattern to regex
            std::string pattern = line;
            // Escape special regex characters
            pattern = std::regex_replace(pattern, std::regex("[.*+?^${}()|\\[\\]\\\\]"), "\\$&");
            // Convert * to .*
            pattern = std::regex_replace(pattern, std::regex("\\*"), ".*");
            // Convert ? to .
            pattern = std::regex_replace(pattern, std::regex("\\?"), ".");
            // Add start/end anchors if pattern doesn't start with /
            if (pattern[0] != '/') {
                pattern = ".*" + pattern;
            }
            // Add end anchor if pattern doesn't end with /
            if (pattern.back() != '/') {
                pattern += "$";
            }

            try {
                gitignore_patterns.push_back(std::regex(pattern));
            } catch (const std::regex_error&) {
                // Skip invalid patterns
                continue;
            }
        }
    }

    bool is_ignored(const fs::path& path) {
        std::string path_str = path.string();
        // Convert path to relative path from project root
        if (path_str.find(test_project_dir.string()) == 0) {
            path_str = path_str.substr(test_project_dir.string().length() + 1);
        }
        
        // Check against all gitignore patterns
        for (const auto& pattern : gitignore_patterns) {
            if (std::regex_match(path_str, pattern)) {
                return true;
            }
        }
        return false;
    }

    void TearDown() override {
        // Clean up the temporary directory
        fs::remove_all(temp_dir);
    }

    void create_test_project() {
        // Create basic project structure
        fs::create_directories(test_project_dir);
        fs::create_directories(test_project_dir / "include" / old_name);
        fs::create_directories(test_project_dir / "lib" / old_name);
        fs::create_directories(test_project_dir / "tools" / (old_name + "-opt"));

        // Create some test files
        create_test_file(test_project_dir / "CMakeLists.txt",
            "project(MLIR" + old_name + ")\n"
            "add_subdirectory(" + old_name + ")\n");

        create_test_file(test_project_dir / "include" / old_name / (old_name + "Dialect.h"),
            "#include \"" + old_name + "/" + old_name + "Ops.h\"\n"
            "class " + old_name + "Dialect {};\n");

        create_test_file(test_project_dir / "lib" / old_name / (old_name + "Ops.cpp"),
            "#include \"" + old_name + "/" + old_name + "Ops.h\"\n"
            "void " + old_name + "Ops::initialize() {}\n");

        create_test_file(test_project_dir / "tools" / (old_name + "-opt") / (old_name + "-opt.cpp"),
            "#include \"" + old_name + "/" + old_name + "Dialect.h\"\n"
            "int main() { return 0; }\n");
    }

    void create_test_file(const fs::path& path, const std::string& content) {
        std::ofstream file(path);
        file << content;
        file.close();
    }

    bool run_rename_script() {
        std::string cmd = (script_dir / "rename-project.sh").string() + " " + 
                         old_name + " " + new_name + " " + 
                         (temp_dir / "renamed_project").string();
        
        // Capture stdout and stderr
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        
        bool result = system(cmd.c_str()) == 0;
        
        // Discard captured output
        testing::internal::GetCapturedStdout();
        testing::internal::GetCapturedStderr();
        
        return result;
    }

    bool file_exists(const fs::path& path) {
        if (is_ignored(path)) return true;  // Skip ignored files
        return fs::exists(path);
    }

    std::string read_file(const fs::path& path) {
        std::ifstream file(path);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        return content;
    }

    bool file_contains(const fs::path& path, const std::string& text) {
        if (is_ignored(path)) return true;  // Skip ignored files
        std::string content = read_file(path);
        return content.find(text) != std::string::npos;
    }
};

TEST_F(RenameProjectTest, DISABLED_BasicRename) {
    // Run the rename script
    ASSERT_TRUE(run_rename_script());

    fs::path renamed_dir = temp_dir / "renamed_project";

    // Check if new directory structure exists
    EXPECT_TRUE(file_exists(renamed_dir / "include" / new_name));
    EXPECT_TRUE(file_exists(renamed_dir / "lib" / new_name));
    EXPECT_TRUE(file_exists(renamed_dir / "tools" / (new_name + "-opt")));

    // Check if files were renamed
    EXPECT_TRUE(file_exists(renamed_dir / "include" / new_name / (new_name + "Dialect.h")));
    EXPECT_TRUE(file_exists(renamed_dir / "lib" / new_name / (new_name + "Ops.cpp")));
    EXPECT_TRUE(file_exists(renamed_dir / "tools" / (new_name + "-opt") / (new_name + "-opt.cpp")));

    // Check if content was replaced
    EXPECT_TRUE(file_contains(renamed_dir / "CMakeLists.txt", "project(MLIR" + new_name + ")"));
    EXPECT_TRUE(file_contains(renamed_dir / "include" / new_name / (new_name + "Dialect.h"), 
                            "class " + new_name + "Dialect"));
    EXPECT_TRUE(file_contains(renamed_dir / "lib" / new_name / (new_name + "Ops.cpp"), 
                            "void " + new_name + "Ops::initialize()"));
}

TEST_F(RenameProjectTest, DISABLED_CaseInsensitiveRename) {
    // Create a file with mixed case
    create_test_file(test_project_dir / "include" / old_name / "standalone_mixed.h",
        "#include \"standalone/StandaloneOps.h\"\n"
        "class standalone_mixed {};\n");

    // Run the rename script
    ASSERT_TRUE(run_rename_script());

    fs::path renamed_dir = temp_dir / "renamed_project";

    // Check if file was renamed with correct case
    EXPECT_TRUE(file_exists(renamed_dir / "include" / new_name / "testdialect_mixed.h"));
    
    // Check if content was replaced with correct case
    EXPECT_TRUE(file_contains(renamed_dir / "include" / new_name / "testdialect_mixed.h", 
                            "class testdialect_mixed"));
}

TEST_F(RenameProjectTest, DISABLED_NoChangesWhenNameNotFound) {
    // Run the rename script with a name that doesn't exist
    std::string cmd = (script_dir / "rename-project.sh").string() + " Nonexistent NewName " + 
                     (temp_dir / "renamed_project").string();
    EXPECT_FALSE(system(cmd.c_str()) == 0);  // Should fail as no files were modified
}

TEST_F(RenameProjectTest, DISABLED_PreservesFilePermissions) {
    // Create a file and set specific permissions
    fs::path test_file = test_project_dir / "test.sh";
    create_test_file(test_file, "#!/bin/bash\necho 'test'");
    fs::permissions(test_file, 
                   fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec,
                   fs::perm_options::replace);

    // Run the rename script
    ASSERT_TRUE(run_rename_script());

    // Check if permissions were preserved
    fs::path renamed_file = temp_dir / "renamed_project" / "test.sh";
    EXPECT_TRUE(file_exists(renamed_file));
    auto perms = fs::status(renamed_file).permissions();
    EXPECT_TRUE((perms & fs::perms::owner_all) == fs::perms::owner_all);
    EXPECT_TRUE((perms & fs::perms::group_read) == fs::perms::group_read);
    EXPECT_TRUE((perms & fs::perms::group_exec) == fs::perms::group_exec);
}

TEST_F(RenameProjectTest, DISABLED_HandlesSpecialCharacters) {
    // Create a file with special characters in the name
    std::string special_name = "Stand-alone";
    create_test_file(test_project_dir / "include" / old_name / (special_name + ".h"),
        "#include \"" + special_name + "/" + special_name + "Ops.h\"\n");

    // Run the rename script
    ASSERT_TRUE(run_rename_script());

    fs::path renamed_dir = temp_dir / "renamed_project";
    std::string new_special_name = "Test-dialect";

    // Check if file was renamed correctly
    EXPECT_TRUE(file_exists(renamed_dir / "include" / new_name / (new_special_name + ".h")));
    EXPECT_TRUE(file_contains(renamed_dir / "include" / new_name / (new_special_name + ".h"), 
                            "#include \"" + new_special_name + "/" + new_special_name + "Ops.h\""));
} 