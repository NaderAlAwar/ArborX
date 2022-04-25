import kokkos
import pykokkos as pk

import PyArborX
from implementation import Implementation

class Parameters:
    def __init__(self):
        self._print_timers: bool = False
        self._implementation = Implementation.FDBSCAN_DenseBox

    def setPrintTimers(self, print_timers: bool):
        self._print_timers = print_timers
        return self

    def setImplementation(self, impl: Implementation):
        self._implementation = impl
        return self

@pk.workunit
def iota(i: int, v: pk.View1D[int], value: int):
    v[i] = value + i

@pk.workunit
def finalize_labels(i: int, cluster_sizes: pk.View1D[int], labels: pk.View1D[int]):
    vstat: int = labels[i]
    old: int = vstat
    next_label: int = labels[vstat]

    while vstat > next_label:
        vstat = next_label
        next_label = labels[vstat]

    if vstat != old:
        labels[i] = vstat

    pk.atomic_increment(cluster_sizes, [labels[i]])

@pk.workunit
def mark_noise(i: int, cluster_sizes: pk.View1D[int], labels: pk.View1D[int]):
    if cluster_sizes[labels[i]] == 1:
        labels[i] = -1

class PrimitivesWithRadius:
    def __init__(self, primitives, eps):
        self.primitives = primitives
        self.eps = eps


def dbscan(exec_space, primitives, eps: float, core_min_size: int, parameters = Parameters()):
    assert(eps > 0)
    assert(core_min_size >= 2)

    is_special_case: bool = (core_min_size == 2)
    n: int = primitives.extent(0)

    num_neigh = pk.View([0], dtype=int)
    labels = pk.View([n], dtype=int)

    pk.parallel_for(n, iota, v=labels, value=0)

    if parameters._implementation is Implementation.FDBSCAN:
        bvh = PyArborX.BVH(exec_space, primitives.array)
        predicates = PyArborX.PrimitivesWithRadius(primitives.array, eps)

        if is_special_case:
            core_points = PyArborX.CCSCorePoints()
            callback = PyArborX.FDBSCANCallback(labels.array, core_points)
            bvh.query(exec_space, predicates, callback)

    cluster_sizes = pk.View([n], dtype=int)
    pk.parallel_for(pk.RangePolicy(pk.Cuda, 0, n), finalize_labels, cluster_sizes=cluster_sizes, labels=labels)
    pk.parallel_for(pk.RangePolicy(pk.Cuda, 0, n), mark_noise, cluster_sizes=cluster_sizes, labels=labels)

    return labels