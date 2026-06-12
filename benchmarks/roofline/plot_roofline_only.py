import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Measured roofline values on NVIDIA TITAN V
# -----------------------------------------------------------------------------
PEAK_FP32_GFLOPS = 12461.1
PEAK_MEM_BW_GBPS = 539.9

RIDGE_POINT = PEAK_FP32_GFLOPS / PEAK_MEM_BW_GBPS

# -----------------------------------------------------------------------------
# Roofline curve
# -----------------------------------------------------------------------------
x_min = 0.1
x_max = 2000.0

ai_values = np.logspace(np.log10(x_min), np.log10(x_max), 500)

memory_roof = ai_values * PEAK_MEM_BW_GBPS
compute_roof = np.full_like(ai_values, PEAK_FP32_GFLOPS)

roofline = np.minimum(memory_roof, compute_roof)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 7))

plt.loglog(
    ai_values,
    roofline,
    linewidth=2.8,
    label="Estimated Roofline"
)

plt.loglog(
    ai_values,
    memory_roof,
    linestyle="--",
    linewidth=1.4,
    label=f"Memory Roof: {PEAK_MEM_BW_GBPS:.1f} GB/s"
)

plt.axhline(
    PEAK_FP32_GFLOPS,
    linestyle="--",
    linewidth=1.4,
    label=f"Compute Roof: {PEAK_FP32_GFLOPS:.1f} GFLOPS"
)

plt.axvline(
    RIDGE_POINT,
    linestyle=":",
    linewidth=1.8,
    label=f"Ridge Point: {RIDGE_POINT:.1f} FLOP/byte"
)

plt.scatter(
    RIDGE_POINT,
    PEAK_FP32_GFLOPS,
    s=70,
    zorder=5
)

plt.annotate(
    f"Ridge Point\n{RIDGE_POINT:.1f} FLOP/byte",
    xy=(RIDGE_POINT, PEAK_FP32_GFLOPS),
    xytext=(12, -35),
    textcoords="offset points",
    fontsize=10
)

plt.title("Estimated FP32 Roofline on NVIDIA TITAN V")
plt.xlabel("Arithmetic Intensity [FLOP/byte]")
plt.ylabel("Performance [GFLOPS]")

plt.grid(True, which="both", linestyle=":", linewidth=0.7)
plt.legend()
plt.tight_layout()

output_path = "roofline_titan_v_fp32_only.png"
plt.savefig(output_path, dpi=200)

print(f"Saved roofline graph to: {output_path}")
print()
print("Roofline values")
print(f"  Peak FP32      : {PEAK_FP32_GFLOPS:.1f} GFLOPS")
print(f"  Peak bandwidth : {PEAK_MEM_BW_GBPS:.1f} GB/s")
print(f"  Ridge point    : {RIDGE_POINT:.2f} FLOP/byte")
