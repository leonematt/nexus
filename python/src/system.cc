#include "pynexus.h"
#include <iostream>

namespace py = pybind11;

// Function to get a value from the dictionary
py::object pynexus::create_buffer(nxs_int size, const std::string& key) {
    return py::none();  // Return None if key is not found
}

