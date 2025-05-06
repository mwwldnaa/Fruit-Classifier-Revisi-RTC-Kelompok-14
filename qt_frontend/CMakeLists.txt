cmake_minimum_required(VERSION 3.16)
project(FruitClassifier)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)

set(RUST_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../rust_backend/target/release")
set(RUST_LIBRARY_NAME "fruit_classifier")
set(RUST_LIBRARY_FULL_PATH "${RUST_LIBRARY_DIR}/lib${RUST_LIBRARY_NAME}.so")

add_library(rust_library SHARED IMPORTED)
set_target_properties(rust_library PROPERTIES
    IMPORTED_LOCATION ${RUST_LIBRARY_FULL_PATH}
)

add_executable(FruitClassifier
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.h
    src/plotwidget.cpp
    src/plotwidget.h
)

target_include_directories(FruitClassifier PRIVATE
    ${INCLUDE_DIR}
)

target_link_libraries(FruitClassifier
    Qt${QT_VERSION_MAJOR}::Widgets
    rust_library
)

# Copy dataset folder to build directory
file(COPY ${CMAKE_SOURCE_DIR}/dataset DESTINATION ${CMAKE_BINARY_DIR})
