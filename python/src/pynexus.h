#ifndef PYNEXUS_H
#define PYNEXUS_H

#include <nexus-api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pynexus {

void init_system_bindings(py::module &m);

}  // namespace pynexus

#endif  // PYNEXUS_H
