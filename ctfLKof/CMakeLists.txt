project( ctfLKof )
find_package( OpenCV REQUIRED )
add_executable( ctfLKof ctfLKof )
target_link_libraries (ctfLKof ${OpenCV_LIBS})