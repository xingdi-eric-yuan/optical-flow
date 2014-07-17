project( LKof )
find_package( OpenCV REQUIRED )
add_executable( LKof LKof )
target_link_libraries (LKof ${OpenCV_LIBS})