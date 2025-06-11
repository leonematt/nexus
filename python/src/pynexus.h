#ifndef PYNEXUS_H
#define PYNEXUS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nexus-api.h>

namespace py = pybind11;

namespace pynexus {

    void init_system_bindings(py::module &m);

// Function to read from a Python dictionary
py::object create_buffer(nxs_int size, const std::string &key = "");

// Function to read from a Python dictionary
void read_dict(const py::dict& dict);

// Function to modify a Python dictionary
void write_dict(py::dict& dict);

}

#endif // PYNEXUS_H
