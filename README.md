# rdma-shim

This library provides the essential CUDA implementation to do RDMA operations on a mlx5-supported card.
This needs a custom version of `rdma-core` which exposes some internals, available [here](https://github.com/rdma-from-gpu/rdma-core).

# License

(C) 2023-2024 Massimo Girondi girondi@kth.se

(C) 2023 Mariano Scazzariello marianos@kth.se

The original code in this repository is licensed under the GNU GPL v3 license. See [LICENSE](LICENSE).

Some minor portions of the source code have been borrowed from [NVIDIA NVSHMEM](https://docs.nvidia.com/nvshmem/api/introduction.html) and [UCX](https://github.com/openucx/ucx).
Such portions are covered by the original licenses.




