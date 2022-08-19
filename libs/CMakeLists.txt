find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
# Build glad library
add_library(glad STATIC glad/src/glad.c glad/include/glad/glad.h glad/include/KHR/khrplatform.h)
target_include_directories(glad PUBLIC glad/include)
target_link_libraries(glad PUBLIC OpenGL::OpenGL)
add_library(glad_egl STATIC glad_egl/src/glad_egl.c glad_egl/include/glad/glad_egl.h glad_egl/include/KHR/khrplatform.h)
target_include_directories(glad_egl PUBLIC glad_egl/include)
target_link_libraries(glad PUBLIC OpenGL::EGL)

# Build imgui / implot library

add_library(imgui STATIC
        imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_widgets.cpp imgui/imgui_demo.cpp
        imgui/examples/imgui_impl_glfw.cpp imgui/examples/imgui_impl_opengl3.cpp
        implot/implot.cpp implot/implot_demo.cpp implot/implot_items.cpp)
target_include_directories(imgui PUBLIC imgui implot)
target_link_libraries(imgui PUBLIC glfw glad)
