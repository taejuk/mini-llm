"""
Theoretical Roofline Model — From Hardware Spec to Ceiling

Roofline (Williams 2009):
    P(I) = min(P_peak, BW * I)               [FLOPS]
    ridge point I* = P_peak / BW             [FLOPs/Byte]

  - I < I*  : memory-bound region   (slope = BW, log-log 에서 기울기 1)
  - I > I*  : compute-bound region  (수평선, height = P_peak)
"""

from __future__ import annotations
import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 0. Compute Capability lookup
#    cudaGetDeviceProperties 가 *주지 않는* 값들:
#      - SM 당 FP32/FP64 core 개수
#      - SM 당 Tensor Core 개수, TC 의 cycle 당 FMA 처리량
#    NVIDIA tuning guide / whitepaper 에 흩어져 있어서 lookup 으로 보강.
# ──────────────────────────────────────────────────────────────────────

#   (major, minor) → dict(fp32, fp64, tc, tc_fma)
#     fp32     : FP32 cores per SM
#     fp64     : FP64 cores per SM (consumer 카드는 매우 적음)
#     tc       : Tensor Cores per SM (Volta+; 0 이면 TC 없음)
#     tc_fma   : TC 의 cycle 당 FP16-acc-FP32 FMA 개수
CC_TABLE: Dict[Tuple[int, int], Dict[str, int]] = {
    (6, 0): dict(fp32=64,  fp64=32, tc=0, tc_fma=0),     # P100
    (6, 1): dict(fp32=128, fp64=4,  tc=0, tc_fma=0),     # GP102/4 (1080 Ti)
    (7, 0): dict(fp32=64,  fp64=32, tc=8, tc_fma=64),    # V100 / TITAN V
    (7, 5): dict(fp32=64,  fp64=2,  tc=8, tc_fma=64),    # Turing (T4, 2080)
    (8, 0): dict(fp32=64,  fp64=32, tc=4, tc_fma=256),   # A100
    (8, 6): dict(fp32=128, fp64=2,  tc=4, tc_fma=128),   # GA10x (3090)
    (8, 9): dict(fp32=128, fp64=2,  tc=4, tc_fma=256),   # AD10x (4090)
    (9, 0): dict(fp32=128, fp64=64, tc=4, tc_fma=512),   # H100
}


# ──────────────────────────────────────────────────────────────────────
# 1. 하드웨어 스펙 컨테이너
#    벤더 whitepaper / nvidia-smi -q / deviceQuery 에서 그대로 옮겨옴
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HardwareSpec:
    """GPU 의 raw 하드웨어 파라미터. peak 값은 여기서 *계산* 한다."""
    # ── 필수(default 없음) ──────────────────────────────────────────
    name: str
    sm_count: int                       # SM 개수
    boost_clock_mhz: float              # SM clock (MHz)
    mem_bus_width_bit: int              # HBM/GDDR bus width (bit)
    mem_clock_mhz: float                # memory clock (MHz, raw for HBM)

    # ── Compute side defaults ──────────────────────────────────────
    fp32_cores_per_sm: int = 64         # FMA-capable FP32 lane / SM
    fp64_cores_per_sm: int = 32         # FP64 lane / SM (server 칩 기준)
    tensor_cores_per_sm: int = 8        # Tensor core 개수 / SM (Volta+)
    tc_fp16_fma_per_cycle: int = 64     # TC FP16 FMA / TC / cycle
                                        #   Volta=64, Ampere=256, Hopper=512

    # ── Memory side defaults ───────────────────────────────────────
    mem_data_rate_per_clock: int = 2    # GDDR=2 (DDR), HBM2/3 = 2
    l2_bandwidth_gbs: float | None = None
    smem_banks_per_sm: int = 32         # 32 banks × 4B/cycle/bank

    # ──────────────────────────────────────────────────────────────
    # Factory: gpu_probe.cu 가 만든 JSON 에서 spec 생성
    # ──────────────────────────────────────────────────────────────
    @classmethod
    def from_probe_json(cls, path: str | Path) -> "HardwareSpec":
        """
        `gpu_probe` 가 출력한 JSON 파일을 읽어 HardwareSpec 생성.

        deviceProp 에 없는 값(FP32 core/SM, TC/SM, TC FMA/cycle)은
        compute capability 로 CC_TABLE 에서 보강한다. 해당 CC 가
        테이블에 없으면 Volta(7.0) 값으로 fallback 하고 경고 출력.
        """
        with open(path) as f:
            j = json.load(f)

        cc = (j["compute_capability_major"], j["compute_capability_minor"])
        if cc not in CC_TABLE:
            print(f"  ! warning: compute capability {cc[0]}.{cc[1]} not in "
                  f"CC_TABLE — falling back to (7,0). Edit CC_TABLE for "
                  f"accuracy on this chip.")
            cc = (7, 0)
        caps = CC_TABLE[cc]

        return cls(
            name              = j["name"],
            sm_count          = j["sm_count"],
            boost_clock_mhz   = j["clock_rate_khz"] / 1e3,
            mem_bus_width_bit = j["mem_bus_width_bit"],
            mem_clock_mhz     = j["mem_clock_rate_khz"] / 1e3,
            fp32_cores_per_sm     = caps["fp32"],
            fp64_cores_per_sm     = caps["fp64"],
            tensor_cores_per_sm   = caps["tc"],
            tc_fp16_fma_per_cycle = caps["tc_fma"],
            # l2_bandwidth_gbs 는 CUDA 가 노출하지 않음 → 사용자가 직접 설정
        )

    @classmethod
    def from_probe_binary(cls, binary: str | Path = "./gpu_probe",
                          device: int = 0) -> "HardwareSpec":
        """`gpu_probe` 실행 파일을 직접 호출(JSON 파일을 거치지 않음)."""
        out = subprocess.check_output([str(binary), str(device)], text=True)
        j = json.loads(out)
        # 임시 dict 로 재활용
        tmp = Path("/tmp/_gpu_probe_tmp.json")
        tmp.write_text(out)
        return cls.from_probe_json(tmp)


# ──────────────────────────────────────────────────────────────────────
# 2. Peak 계산식 — roofline 의 "천장" 들
# ──────────────────────────────────────────────────────────────────────

def peak_fp32_gflops(spec: HardwareSpec) -> float:
    """
    Peak FP32 = SM × cores/SM × clock × 2 (FMA = 2 FLOP)
    단위: GFLOPS
    """
    flops_per_cycle = spec.sm_count * spec.fp32_cores_per_sm * 2
    return flops_per_cycle * spec.boost_clock_mhz * 1e-3   # MHz×… → GFLOPS


def peak_fp64_gflops(spec: HardwareSpec) -> float:
    """Peak FP64 = SM × FP64 cores/SM × clock × 2"""
    flops_per_cycle = spec.sm_count * spec.fp64_cores_per_sm * 2
    return flops_per_cycle * spec.boost_clock_mhz * 1e-3


def peak_tensor_fp16_gflops(spec: HardwareSpec) -> float:
    """
    Peak Tensor Core FP16 (FP16 multiply, FP32 accumulate)
      = SM × TC/SM × FMA/cycle × clock × 2
    """
    flops_per_cycle = (spec.sm_count
                       * spec.tensor_cores_per_sm
                       * spec.tc_fp16_fma_per_cycle
                       * 2)
    return flops_per_cycle * spec.boost_clock_mhz * 1e-3


def peak_hbm_bandwidth_gbs(spec: HardwareSpec) -> float:
    """
    Peak DRAM BW = bus_width(bit)/8 × mem_clock(Hz) × data_rate
    단위: GB/s

    예) TITAN V HBM2: 3072 bit / 8 × 850 MHz × 2 = 652.8 GB/s
    """
    bytes_per_bus_cycle = spec.mem_bus_width_bit / 8
    return (bytes_per_bus_cycle
            * spec.mem_clock_mhz
            * spec.mem_data_rate_per_clock
            * 1e-3)  # MHz → GHz → GB/s


def peak_smem_bandwidth_gbs(spec: HardwareSpec) -> float:
    """
    Shared memory BW = SM × banks × 4 B/cycle × clock
      (Volta+ : 32 banks × 4B/bank/cycle = 128 B/SM/cycle)
    """
    bytes_per_sm_per_cycle = spec.smem_banks_per_sm * 4
    return (spec.sm_count
            * bytes_per_sm_per_cycle
            * spec.boost_clock_mhz
            * 1e-3)


# ──────────────────────────────────────────────────────────────────────
# 3. Ridge point — compute / memory 경계
# ──────────────────────────────────────────────────────────────────────

def ridge_point(peak_gflops: float, peak_bw_gbs: float) -> float:
    """
    I* = P_peak / BW   [FLOPs / Byte]
      커널의 산술 강도(arithmetic intensity)가 이 값보다 작으면
      memory-bound, 크면 compute-bound.
    """
    return peak_gflops / peak_bw_gbs


# ──────────────────────────────────────────────────────────────────────
# 4. Roofline 차트 그리기
# ──────────────────────────────────────────────────────────────────────

def attainable_gflops(intensity: np.ndarray,
                      peak_gflops: float,
                      peak_bw_gbs: float) -> np.ndarray:
    """P(I) = min(P_peak, BW × I)"""
    return np.minimum(peak_gflops, peak_bw_gbs * intensity)


def plot_theoretical_roofline(spec: HardwareSpec,
                              savefig: str | None = None) -> None:
    # ── 천장 계산 ─────────────────────────────────────────────────
    fp32  = peak_fp32_gflops(spec)
    fp64  = peak_fp64_gflops(spec)
    tcfp16 = peak_tensor_fp16_gflops(spec)
    bw_hbm = peak_hbm_bandwidth_gbs(spec)
    bw_smem = peak_smem_bandwidth_gbs(spec)
    bw_l2 = spec.l2_bandwidth_gbs  # 알려진 값만

    # ── 콘솔 출력 (스펙 → 숫자 흐름이 보이도록) ───────────────────
    print(f"\n── {spec.name} ─────────────────────────────────")
    print(f"  SM × cores × 2 × clock  =  Peak FP32")
    print(f"  {spec.sm_count} × {spec.fp32_cores_per_sm} × 2 × "
          f"{spec.boost_clock_mhz} MHz = {fp32:>9.1f} GFLOPS")
    print(f"  Peak FP64                              = {fp64:>9.1f} GFLOPS")
    print(f"  Peak Tensor Core FP16                  = {tcfp16:>9.1f} GFLOPS")
    print(f"  HBM bus/8 × clock × 2 = "
          f"{spec.mem_bus_width_bit}/8 × {spec.mem_clock_mhz} × 2 "
          f"= {bw_hbm:>6.1f} GB/s")
    print(f"  Shared-mem aggregate                    = {bw_smem:>6.1f} GB/s")
    print(f"  Ridge point (FP32/HBM)                 = "
          f"{ridge_point(fp32, bw_hbm):.1f} FLOP/Byte")
    print(f"  Ridge point (TC FP16/HBM)              = "
          f"{ridge_point(tcfp16, bw_hbm):.1f} FLOP/Byte")

    # ── 플롯 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    intensity = np.logspace(-2, 4, 400)   # 0.01 ~ 10000 FLOP/B

    # compute ceilings — 수평선
    ceilings = [("FP64",            fp64,   "#7f7f7f"),
                ("FP32",            fp32,   "#1f77b4"),
                ("Tensor Core FP16", tcfp16, "#d62728")]
    for label, peak, color in ceilings:
        ax.axhline(peak, color=color, lw=1.4, ls="--", alpha=0.8)
        ax.text(intensity[-1] * 0.6, peak * 1.08,
                f"{label}: {peak:,.0f} GFLOPS",
                color=color, fontsize=9, ha="right")

    # memory ceilings — 기울기 1
    bws = [("HBM",  bw_hbm,  "#2ca02c")]
    if bw_l2 is not None:
        bws.append(("L2",  bw_l2,  "#9467bd"))
    bws.append(("Shared", bw_smem, "#ff7f0e"))

    # roofline 자체(가장 큰 compute ceiling 기준)
    top_peak = max(p for _, p, _ in ceilings)
    for label, bw, color in bws:
        y = attainable_gflops(intensity, top_peak, bw)
        ax.plot(intensity, y, color=color, lw=1.8, label=f"{label}: {bw:,.0f} GB/s")

        # ridge point 마커
        I_star = top_peak / bw
        ax.plot(I_star, top_peak, "o", color=color, ms=6)
        ax.annotate(f"I* = {I_star:.1f}",
                    xy=(I_star, top_peak),
                    xytext=(I_star * 1.4, top_peak * 0.35),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.6))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity  [FLOPs / Byte]")
    ax.set_ylabel("Attainable Performance  [GFLOPS]")
    ax.set_title(f"Theoretical Roofline — {spec.name}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, title="Memory ceilings")
    ax.set_ylim(1, top_peak * 5)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
        print(f"\n  saved → {savefig}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────
# 5. 예시: NVIDIA TITAN V (Volta, sm_70) — 사용자 GPU
# ──────────────────────────────────────────────────────────────────────

TITAN_V = HardwareSpec(
    name              = "NVIDIA TITAN V (Volta, sm_70)",
    sm_count          = 80,
    boost_clock_mhz   = 1455,
    fp32_cores_per_sm = 64,
    fp64_cores_per_sm = 32,
    tensor_cores_per_sm    = 8,
    tc_fp16_fma_per_cycle  = 64,    # Volta 세대
    mem_bus_width_bit = 3072,        # HBM2 ×3 stacks × 1024 bit
    mem_clock_mhz     = 850,         # HBM2
    mem_data_rate_per_clock = 2,
    l2_bandwidth_gbs  = 2155,        # 추정 (~3.3× HBM)
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--probe-json", type=str,
                   help="gpu_probe 가 출력한 JSON 파일에서 spec 읽기")
    g.add_argument("--probe-binary", type=str,
                   help="gpu_probe 실행 파일을 직접 호출 (default device 0)")
    g.add_argument("--builtin-titan-v", action="store_true",
                   help="하드코딩된 TITAN_V spec 사용 (GPU 없는 환경 검증용)")
    ap.add_argument("--device", type=int, default=0,
                    help="--probe-binary 와 함께 쓸 device index")
    ap.add_argument("--l2-bw", type=float, default=None,
                    help="L2 bandwidth (GB/s). probe 는 못 가져옴 → 옵션 제공.")
    ap.add_argument("-o", "--output", type=str,
                    default="theoretical_roofline.png",
                    help="저장할 PNG 경로")
    args = ap.parse_args()

    if args.probe_json:
        spec = HardwareSpec.from_probe_json(args.probe_json)
    elif args.probe_binary:
        spec = HardwareSpec.from_probe_binary(args.probe_binary, args.device)
    else:
        spec = TITAN_V    # default (호환 유지)

    if args.l2_bw is not None:
        spec.l2_bandwidth_gbs = args.l2_bw

    plot_theoretical_roofline(spec, savefig=args.output)

