from torch.utils.cpp_extension import load

cuda_module = load(
    name="add2",
    extra_include_paths=["include"],
    sources=["kernel/add2_ops.cpp", "kernel/add2_kernel.cu"],
    verbose=True,
)

cuda_module.torch_launch_add2(cuda_c, a, b, n)
