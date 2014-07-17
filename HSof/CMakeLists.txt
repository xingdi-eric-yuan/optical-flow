project( HSof )
find_package( OpenCV REQUIRED )
add_executable( HSof HSof )
target_link_libraries (HSof ${OpenCV_LIBS})