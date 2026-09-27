find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# An existing MSVC import-library bundle is an alternative to vcpkg. This keeps
# offline Windows builds usable without changing the normal vcpkg path.
set(NINFER_WINDOWS_LIBS_DIR "" CACHE PATH
  "Directory containing MSVC FFmpeg/CURL import libraries and filtered headers")

if(WIN32 AND NINFER_WINDOWS_LIBS_DIR)
  if(NOT EXISTS "${NINFER_WINDOWS_LIBS_DIR}/avformat.lib" OR
     NOT EXISTS "${NINFER_WINDOWS_LIBS_DIR}/libcurl.lib")
    message(FATAL_ERROR
      "NINFER_WINDOWS_LIBS_DIR must contain avformat.lib and libcurl.lib")
  endif()
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE
    "${NINFER_WINDOWS_LIBS_DIR}/ffmpeg-include")
  target_compile_definitions(ninfer_ffmpeg_dependencies INTERFACE __STDC_CONSTANT_MACROS)
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE
    "${NINFER_WINDOWS_LIBS_DIR}/avformat.lib"
    "${NINFER_WINDOWS_LIBS_DIR}/avcodec.lib"
    "${NINFER_WINDOWS_LIBS_DIR}/avutil.lib"
    "${NINFER_WINDOWS_LIBS_DIR}/swscale.lib")
  add_library(ninfer_curl_dependencies INTERFACE)
  target_include_directories(ninfer_curl_dependencies INTERFACE
    "${NINFER_WINDOWS_LIBS_DIR}/curl-include")
  target_link_libraries(ninfer_curl_dependencies INTERFACE
    "${NINFER_WINDOWS_LIBS_DIR}/libcurl.lib")
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
  set(NINFER_CURL_TARGET ninfer_curl_dependencies)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
elseif(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(WIN32 AND NINFER_WINDOWS_LIBS_DIR)
    set(NINFER_CURL_TARGET ninfer_curl_dependencies)
  elseif(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
