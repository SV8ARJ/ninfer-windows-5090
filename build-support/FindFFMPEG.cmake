set(FFMPEG_ROOT "${CMAKE_CURRENT_LIST_DIR}/../windows-libs")

set(FFMPEG_INCLUDE_DIRS "${FFMPEG_ROOT}/ffmpeg-include")
set(FFMPEG_LIBRARY_DIRS "${FFMPEG_ROOT}")
set(FFMPEG_LIBRARIES
  "${FFMPEG_ROOT}/avformat.lib"
  "${FFMPEG_ROOT}/avcodec.lib"
  "${FFMPEG_ROOT}/avutil.lib"
  "${FFMPEG_ROOT}/swscale.lib")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFMPEG DEFAULT_MSG FFMPEG_INCLUDE_DIRS FFMPEG_LIBRARIES)

mark_as_advanced(FFMPEG_INCLUDE_DIRS FFMPEG_LIBRARY_DIRS FFMPEG_LIBRARIES)
