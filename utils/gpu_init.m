function dev = gpu_init()
% GPU_INIT Detects and initializes the GPU for FEM computation.
%   Returns the gpuDevice object after resetting it for a clean state.

    dev = gpuDevice;
    reset(dev);  % Clear any stale GPU memory

    fprintf('\n=== GPU Initialized ===\n');
    fprintf('  Device : %s\n', dev.Name);
    fprintf('  Compute: %d.%d\n', dev.ComputeCapability);
    fprintf('  Memory : %.1f GB (%.1f GB free)\n', ...
        dev.TotalMemory/1e9, dev.AvailableMemory/1e9);
    fprintf('=======================\n\n');
end
