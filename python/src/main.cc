#include <pybind11/pybind11.h>

#include "pynexus.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(libnexus, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: nexus

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    auto sm = m.def_submodule("runtime");
    sm.def("buffer", &pynexus::create_buffer, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def_submodule("devices");

#if 0
    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");
#endif

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
