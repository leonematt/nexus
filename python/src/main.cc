#include <pybind11/pybind11.h>

#include "pynexus.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(libnexus, m) {
  m.doc() = R"pbdoc(
        Nexus - Python API
        -----------------------

        .. currentmodule:: nexus

        .. autosummary::
           :toctree: _generate

           system
           devices
    )pbdoc";

  // remove extra 'system' module (its redundant)
  pynexus::init_system_bindings(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
