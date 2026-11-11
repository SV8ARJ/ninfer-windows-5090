set(CURL_ROOT "${CMAKE_CURRENT_LIST_DIR}/../windows-libs")

set(CURL_INCLUDE_DIRS "${CURL_ROOT}/curl-include")
set(CURL_LIBRARY "${CURL_ROOT}/libcurl.lib")
set(CURL_VERSION_STRING "8.13.0")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CURL
  REQUIRED_VARS CURL_LIBRARY CURL_INCLUDE_DIRS
  VERSION_VAR CURL_VERSION_STRING)

if(CURL_FOUND AND NOT TARGET CURL::libcurl)
  add_library(CURL::libcurl UNKNOWN IMPORTED)
  set_target_properties(CURL::libcurl PROPERTIES
    IMPORTED_LOCATION "${CURL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIRS}")
endif()

mark_as_advanced(CURL_INCLUDE_DIRS CURL_LIBRARY)
