# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.30.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.30.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jinyilu/Desktop/projects/colour

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jinyilu/Desktop/projects/colour/build

# Include any dependencies generated for this target.
include CMakeFiles/colour.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/colour.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/colour.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/colour.dir/flags.make

CMakeFiles/colour.dir/main.cpp.o: CMakeFiles/colour.dir/flags.make
CMakeFiles/colour.dir/main.cpp.o: /Users/jinyilu/Desktop/projects/colour/main.cpp
CMakeFiles/colour.dir/main.cpp.o: CMakeFiles/colour.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jinyilu/Desktop/projects/colour/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/colour.dir/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/colour.dir/main.cpp.o -MF CMakeFiles/colour.dir/main.cpp.o.d -o CMakeFiles/colour.dir/main.cpp.o -c /Users/jinyilu/Desktop/projects/colour/main.cpp

CMakeFiles/colour.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/colour.dir/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jinyilu/Desktop/projects/colour/main.cpp > CMakeFiles/colour.dir/main.cpp.i

CMakeFiles/colour.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/colour.dir/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jinyilu/Desktop/projects/colour/main.cpp -o CMakeFiles/colour.dir/main.cpp.s

# Object files for target colour
colour_OBJECTS = \
"CMakeFiles/colour.dir/main.cpp.o"

# External object files for target colour
colour_EXTERNAL_OBJECTS =

colour: CMakeFiles/colour.dir/main.cpp.o
colour: CMakeFiles/colour.dir/build.make
colour: /usr/local/lib/libsfml-graphics.2.6.1.dylib
colour: /usr/local/lib/libsfml-audio.2.6.1.dylib
colour: /usr/local/lib/libsfml-network.2.6.1.dylib
colour: /usr/local/lib/libsfml-window.2.6.1.dylib
colour: /usr/local/lib/libsfml-system.2.6.1.dylib
colour: CMakeFiles/colour.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/jinyilu/Desktop/projects/colour/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable colour"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/colour.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/colour.dir/build: colour
.PHONY : CMakeFiles/colour.dir/build

CMakeFiles/colour.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/colour.dir/cmake_clean.cmake
.PHONY : CMakeFiles/colour.dir/clean

CMakeFiles/colour.dir/depend:
	cd /Users/jinyilu/Desktop/projects/colour/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jinyilu/Desktop/projects/colour /Users/jinyilu/Desktop/projects/colour /Users/jinyilu/Desktop/projects/colour/build /Users/jinyilu/Desktop/projects/colour/build /Users/jinyilu/Desktop/projects/colour/build/CMakeFiles/colour.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/colour.dir/depend

