# the minimum version of CMake.
cmake_minimum_required(VERSION 3.4.1)
project(TestXComponent)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS ${OHOS_ARCH})

if (OHOS_ARCH MATCHES "armeabi-v7a")
    set(LIB_PATH ${NATIVERENDER_ROOT_PATH}/3rd/lib32)
elseif (OHOS_ARCH MATCHES "arm64-v8a")
    set(LIB_PATH ${NATIVERENDER_ROOT_PATH}/3rd/lib64)
endif()

include_directories(
    ${NATIVERENDER_ROOT_PATH}
    ${NATIVERENDER_ROOT_PATH}/include
    ${LIB_PATH}/aravis-0.8/include
    ${LIB_PATH}/opencv/include
    ${LIB_PATH}/exiv2/include
    ${LIB_PATH}/glib
    ${LIB_PATH}/glib/glib
    ${LIB_PATH}/glib/gio
    ${LIB_PATH}/glib/gmodule
)

add_library(entry SHARED
    hello.cpp
    cameraLinkThreadWorker.cpp
    inferHandle.cpp
    test_infer.cpp
)

# ============ SHARED LIBRARY: entry =============
target_link_libraries(entry PUBLIC
    libace_napi.z.so
    libace_ndk.z.so
    libnative_window.so
    libhilog_ndk.z.so
    libmindspore_lite_ndk.so
    librawfile.z.so
    ${LIB_PATH}/aravis-0.8/libs/libaravis.z.so
    ${LIB_PATH}/aravis-0.8/libs/libshared_libz.z.so
    ${LIB_PATH}/aravis-0.8/libs/libusb_shared.z.so
    ${LIB_PATH}/aravis-0.8/libs/libxml2.z.so
    ${LIB_PATH}/aravis-0.8/libs/libglibpcre.z.so
    #${LIB_PATH}/aravis-0.8/libs/libgio.z.so
    ${LIB_PATH}/aravis-0.8/libs/libglib.z.so
    ${LIB_PATH}/aravis-0.8/libs/libgmodule.z.so
    ${LIB_PATH}/aravis-0.8/libs/libgobject.z.so
    ${LIB_PATH}/opencv/libs/libopencv_core.so
    ${LIB_PATH}/opencv/libs/libopencv_imgproc.so
    ${LIB_PATH}/opencv/libs/libopencv_imgcodecs.so
    ${LIB_PATH}/exiv2/libs/libexiv2.so
)

# ============ EXECUTABLE: test_infer =============
add_executable(test_infer
    test_infer.cpp
    inferHandle.cpp
)

target_link_libraries(test_infer
    ${LIB_PATH}/aravis-0.8/libs/libaravis.z.so
    ${LIB_PATH}/aravis-0.8/libs/libshared_libz.z.so
    ${LIB_PATH}/aravis-0.8/libs/libusb_shared.z.so
    ${LIB_PATH}/aravis-0.8/libs/libxml2.z.so
    ${LIB_PATH}/aravis-0.8/libs/libglibpcre.z.so
    #${LIB_PATH}/aravis-0.8/libs/libgio.z.so
    ${LIB_PATH}/aravis-0.8/libs/libglib.z.so
    ${LIB_PATH}/aravis-0.8/libs/libgmodule.z.so
    ${LIB_PATH}/aravis-0.8/libs/libgobject.z.so
    ${LIB_PATH}/opencv/libs/libopencv_core.so
    ${LIB_PATH}/opencv/libs/libopencv_imgproc.so
    ${LIB_PATH}/opencv/libs/libopencv_imgcodecs.so
    ${LIB_PATH}/exiv2/libs/libexiv2.so
    libmindspore_lite_ndk.so
    libhilog_ndk.z.so

)
