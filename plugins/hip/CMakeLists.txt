#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/runtime_libs")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)


add_library(hip_plugin hip_runtime.c)

if(UNIX AND (NOT APPLE))
  target_link_options(hip_plugin PRIVATE "LINKER:--exclude-libs,ALL")
endif()
if(WIN32)
  target_link_options(hip_plugin PRIVATE "LINKER:--enable-auto-import,--enable-runtime-pseudo-reloc")
endif()

