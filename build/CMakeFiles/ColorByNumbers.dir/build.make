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
include CMakeFiles/ColorByNumbers.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ColorByNumbers.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ColorByNumbers.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ColorByNumbers.dir/flags.make

CMakeFiles/ColorByNumbers.dir/main.cpp.o: CMakeFiles/ColorByNumbers.dir/flags.make
CMakeFiles/ColorByNumbers.dir/main.cpp.o: /Users/jinyilu/Desktop/projects/colour/main.cpp
CMakeFiles/ColorByNumbers.dir/main.cpp.o: CMakeFiles/ColorByNumbers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jinyilu/Desktop/projects/colour/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ColorByNumbers.dir/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ColorByNumbers.dir/main.cpp.o -MF CMakeFiles/ColorByNumbers.dir/main.cpp.o.d -o CMakeFiles/ColorByNumbers.dir/main.cpp.o -c /Users/jinyilu/Desktop/projects/colour/main.cpp

CMakeFiles/ColorByNumbers.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ColorByNumbers.dir/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jinyilu/Desktop/projects/colour/main.cpp > CMakeFiles/ColorByNumbers.dir/main.cpp.i

CMakeFiles/ColorByNumbers.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ColorByNumbers.dir/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jinyilu/Desktop/projects/colour/main.cpp -o CMakeFiles/ColorByNumbers.dir/main.cpp.s

# Object files for target ColorByNumbers
ColorByNumbers_OBJECTS = \
"CMakeFiles/ColorByNumbers.dir/main.cpp.o"

# External object files for target ColorByNumbers
ColorByNumbers_EXTERNAL_OBJECTS =

ColorByNumbers: CMakeFiles/ColorByNumbers.dir/main.cpp.o
ColorByNumbers: CMakeFiles/ColorByNumbers.dir/build.make
ColorByNumbers: /usr/local/lib/libsfml-graphics.2.6.1.dylib
ColorByNumbers: /usr/local/lib/libopencv_gapi.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_stitching.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_alphamat.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_aruco.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_bgsegm.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_bioinspired.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_ccalib.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_dnn_objdetect.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_dnn_superres.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_dpm.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_face.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_freetype.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_fuzzy.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_hfs.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_img_hash.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_intensity_transform.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_line_descriptor.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_mcc.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_quality.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_rapid.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_reg.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_rgbd.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_saliency.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_sfm.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_signal.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_stereo.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_structured_light.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_superres.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_surface_matching.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_tracking.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_videostab.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_viz.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_wechat_qrcode.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_xfeatures2d.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_xobjdetect.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_xphoto.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libsfml-window.2.6.1.dylib
ColorByNumbers: /usr/local/lib/libsfml-system.2.6.1.dylib
ColorByNumbers: /usr/local/lib/libopencv_shape.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_highgui.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_datasets.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_plot.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_text.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_ml.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_phase_unwrapping.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_optflow.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_ximgproc.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_video.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_videoio.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_imgcodecs.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_objdetect.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_calib3d.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_dnn.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_features2d.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_flann.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_photo.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_imgproc.4.10.0.dylib
ColorByNumbers: /usr/local/lib/libopencv_core.4.10.0.dylib
ColorByNumbers: CMakeFiles/ColorByNumbers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/jinyilu/Desktop/projects/colour/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ColorByNumbers"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ColorByNumbers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ColorByNumbers.dir/build: ColorByNumbers
.PHONY : CMakeFiles/ColorByNumbers.dir/build

CMakeFiles/ColorByNumbers.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ColorByNumbers.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ColorByNumbers.dir/clean

CMakeFiles/ColorByNumbers.dir/depend:
	cd /Users/jinyilu/Desktop/projects/colour/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jinyilu/Desktop/projects/colour /Users/jinyilu/Desktop/projects/colour /Users/jinyilu/Desktop/projects/colour/build /Users/jinyilu/Desktop/projects/colour/build /Users/jinyilu/Desktop/projects/colour/build/CMakeFiles/ColorByNumbers.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/ColorByNumbers.dir/depend

