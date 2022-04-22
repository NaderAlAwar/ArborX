import struct
from typing import List

import kokkos
import pykokkos as pk

import ArborX_DBSCAN
import PyArborX

def loadData(filename: str, binary: bool = True, max_num_points: int = -1):
    mode_print: str = "binary" if binary else "text"
    print(f"Reading in {filename} in {mode_print} mode...")

    if binary:
        with open(filename, "rb") as f:
            contents = f.read()
            info = struct.unpack("i" * ((8) // 4), contents[:8])
            data = struct.unpack("f" * ((len(contents)- 8) // 4), contents[8:])
    else:
        with open(filename, "rb") as f:
            info: List[str] = f.readline().strip().split(" ")
            data = f.readlines()

    num_points: int = int(info[0])
    dim: int = int(info[1])

    assert(dim == 2 or dim == 3)

    if max_num_points > 0 and max_num_points < num_points:
        num_points = max_num_points

    v = pk.View([num_points, dim], pk.float)
    if not binary:
        for index, line in enumerate(data):
            point_data: List[float] = [float(f) for f in line.strip().split(" ")]
            for d in range(dim):
                v[index][d] = point_data[d]
    else:
        for index, data in enumerate(data):
            point_id: int = index // dim
            dimension: int = index % dim
            v[point_id][dimension] = data

    print("done")
    print(f"Read in {num_points} {dim}D points")

    return v

@pk.workunit
def exclusiveScanWorkunit(i: int, update: pk.Acc[int], final_pass: bool, _in: pk.View1D[int], _out: pk.View1D[int]):
    in_i: int = _in[i]
    if final_pass:
        _out[i] = update
    update += in_i


def exclusivePrefixSumTwoViews(space: pk.ExecutionSpace, src: pk.View1D[int], dst: pk.View1D[int]):
    n: int = src.extent(0)
    assert(n == dst.extent(0))

    policy = pk.RangePolicy(space, 0, n)
    pk.parallel_scan(policy, exclusiveScanWorkunit, _in=src, _out=dst)


def exclusivePrefixSum(space: pk.ExecutionSpace, v: pk.View1D[int]) -> None:
    exclusivePrefixSumTwoViews(space, v, v)


@pk.workunit
def computeClusterSizes(i: int, labels: pk.View1D[int], cluster_sizes: pk.View1D[int]):
    if labels[i] < 0:
        return

    pk.atomic_fetch_add(cluster_sizes, [labels[i]], 1)

@pk.workunit
def computeClusterOffsetWithFilter(
    i: int, update: pk.Acc[int], final_pass: bool, cluster_sizes: pk.View1D[int],
    cluster_offset: pk.View1D[int], map_cluster_to_offset_position: pk.View1D[int],
    cluster_min_size: int
):
    is_cluster_too_small: bool = cluster_sizes[i] < cluster_min_size
    if not is_cluster_too_small:
        if final_pass:
            cluster_offset[update] = cluster_sizes[i]
            map_cluster_to_offset_position[i] = update
        update += 1
    else:
        if final_pass:
            map_cluster_to_offset_position[i] = -1

@pk.workunit
def computeClusterIndices(
    i: int, labels: pk.View1D[int], map_cluster_to_offset_position: pk.View1D[int],
    cluster_starts: pk.View1D[int], cluster_indices: pk.View1D[int]
):
    if labels[i] < 0:
        return

    offset_pos: int = map_cluster_to_offset_position[labels[i]]
    if offset_pos != -1:
        position: int = pk.atomic_fetch_add(cluster_starts, [offset_pos], 1)
        cluster_indices[position] = i

def sortAndFilterClusters(
    exec_space: pk.ExecutionSpace, labels: pk.View1D[int],
    cluster_indices: pk.View1D[int], cluster_offset: pk.View1D[int],
    cluster_min_size: int = 1
) -> None:
    assert(cluster_min_size >= 1)

    n: int = labels.extent(0)
    cluster_sizes: pk.View1D[int] = pk.View([n], int)
    pk.parallel_for(pk.RangePolicy(exec_space, 0, n), computeClusterSizes, labels=labels, cluster_sizes=cluster_sizes)

    map_cluster_to_offset_position: pk.View1D[int] = cluster_sizes
    cluster_offset.resize(0, n + 1)

    num_clusters: int = pk.parallel_scan(
        pk.RangePolicy(exec_space, 0, n), computeClusterOffsetWithFilter, cluster_sizes=cluster_sizes,
        cluster_offset=cluster_offset, map_cluster_to_offset_position=map_cluster_to_offset_position,
        cluster_min_size=cluster_min_size)

    cluster_offset.resize(0, num_clusters + 1)
    exclusivePrefixSum(exec_space, cluster_offset)

    # instead of clone()
    cluster_starts = pk.View([cluster_offset.extent(0)], int)
    cluster_starts[:] = cluster_offset[:]

    cluster_indices.resize(0, cluster_offset[-1])
    pk.parallel_for(
        pk.RangePolicy(exec_space, 0, n), computeClusterIndices,
        labels=labels, map_cluster_to_offset_position=map_cluster_to_offset_position,
        cluster_starts=cluster_starts, cluster_indices=cluster_indices)


def run():
    pk.set_default_space(pk.Cuda)
    pk.enable_uvm()
    # args go here
    algorithm: str = "dbscan"
    filename: str = "input.txt"
    max_num_points: int = -1
    binary: bool = False
    eps: float = 1
    cluster_min_size: int = 1
    core_min_size: int = 2
    verify: bool = False
    num_samples: int = -1
    filename_labels: str = ""
    print_dbscan_timers: bool = False

    # Print out the runtime parameters
    print(f"algorithm         : {algorithm}")
    if algorithm == "dbscan":
        print(f"eps               : {eps}")
        print(f"cluster min size  : {cluster_min_size}")
        # print("implementation    : %s\n", ss.str().c_str());
        print(f"verify            : {verify}")

    print(f"minpts            : {core_min_size}")
    mode_print: str = "binary" if binary else "text"
    print(f"filename          : {filename} [{mode_print}, max_pts = {max_num_points}]")

    if filename_labels != "":
        print(f"filename [labels] : {filename_labels} [binary]")
    print(f"samples           : {num_samples}")
    print(f"print timers      : {print_dbscan_timers}")

    primitives = loadData(filename, binary, max_num_points)
    exec_space = PyArborX.Cuda()

    labels = ArborX_DBSCAN.dbscan(exec_space, primitives, eps, core_min_size, None)

    cluster_indices = pk.View([0], dtype=int)
    cluster_offset = pk.View([0], dtype=int)

    sortAndFilterClusters(pk.Cuda, labels, cluster_indices, cluster_offset, cluster_min_size)

    num_clusters: int = cluster_offset.extent(0) - 1
    num_cluster_points: int = cluster_indices.extent(0)

    print(f"\n#clusters       : {num_clusters}")
    print(f"#cluster points : {num_cluster_points} [{(100. * num_cluster_points / primitives.extent(0)):.2f}]")


if __name__ == "__main__":
    run()
    kokkos.finalize()