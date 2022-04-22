#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <ArborX_LinearBVH.hpp>
#include <ArborX_DBSCAN.hpp>
#include <ArborX_DetailsFDBSCAN.hpp>

#include <Kokkos_Core.hpp>

namespace py = pybind11;

// BVH template arguments
using ExecutionSpace = Kokkos::Cuda;
// using MemorySpace = ExecutionSpace::memory_space;
using MemorySpace = Kokkos::CudaUVMSpace;
using Primitives = Kokkos::View<float**, Kokkos::LayoutLeft, MemorySpace>;
using Predicates = ArborX::Details::PrimitivesWithRadius<Primitives>;
using Callback = ArborX::Details::FDBSCANCallback<MemorySpace, ArborX::Details::CCSCorePoints>;
using BVH = ArborX::BasicBoundingVolumeHierarchy<MemorySpace>;

// FDBSCANCallback template arguments
using CorePointsType = ArborX::Details::CCSCorePoints;
using FDBSCANCallback = ArborX::Details::FDBSCANCallback<MemorySpace, CorePointsType>;

// PrimitivesWithRadius template arguments
using PrimitivesWithRadius = ArborX::Details::PrimitivesWithRadius<Primitives>;

Kokkos::View<ArborX::Point*, Kokkos::CudaSpace> vec2view(std::vector<ArborX::Point> const &in, std::string const &label = "")
{
  Kokkos::View<ArborX::Point*, Kokkos::CudaSpace> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<ArborX::Point const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
}

PYBIND11_MODULE(PyArborX, m) {
    py::class_<Kokkos::Cuda>(m, "Cuda")
        .def(py::init([](){ Kokkos::initialize(); return Kokkos::Cuda{}; }));

    // Needed to have default policy argument for query
    py::class_<ArborX::Experimental::TraversalPolicy>(m, "TraversalPolicy");

    py::class_<BVH>(m, "BVH")
        .def(py::init<>())
        .def(py::init<Kokkos::Cuda, Primitives>())
        .def("query", &BVH::query<Kokkos::Cuda, PrimitivesWithRadius, const FDBSCANCallback&>,
              py::arg("space"),
              py::arg("predicates"),
              py::arg("callback"),
              py::arg("policy") = ArborX::Experimental::TraversalPolicy());

    py::class_<CorePointsType>(m, "CCSCorePoints")
        .def(py::init<>());

    py::class_<FDBSCANCallback>(m, "FDBSCANCallback")
        .def(py::init<const Kokkos::View<int*, Kokkos::LayoutLeft, MemorySpace>&, CorePointsType>());

    py::class_<PrimitivesWithRadius>(m, "PrimitivesWithRadius")
        .def(py::init<Primitives, double>());

    py::class_<Kokkos::View<ArborX::Point*, Kokkos::CudaSpace>>(m, "Primitives");

    m.def("vec2view", &vec2view);
}