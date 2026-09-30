#!/usr/bin/env python3
"""
iacheck — ¿Qué modelos de IA locales puede correr mi equipo?

Escanea el hardware (CPU, RAM, GPU/VRAM, memoria unificada de Apple Silicon),
consulta Hugging Face y estima qué modelos caben y a qué velocidad aproximada.

Funciona en Windows, macOS y Linux sólo con la librería estándar de Python 3.8+.
(Si `psutil` está instalado se usa para datos más precisos, pero no es necesario.)

Uso:
    iacheck                      # interfaz de terminal (TUI)
    iacheck scan                 # especificaciones + modelos recomendados
    iacheck search "llama 3"     # buscar modelos y ver si son aptos
    iacheck check Qwen/Qwen3-8B  # análisis detallado de un repositorio
    iacheck specs                # sólo especificaciones
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

__version__ = "1.0.0"
HF_API = "https://huggingface.co/api"
HF_WEB = "https://huggingface.co"
GIB = 1024 ** 3

# --------------------------------------------------------------------------- #
# Salida de terminal
# --------------------------------------------------------------------------- #

if os.name == "nt":
    os.system("")  # activa secuencias ANSI en la consola de Windows 10+
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def c(text: str, *codes: str) -> str:
    if not USE_COLOR or not codes:
        return text
    return "\x1b[" + ";".join(codes) + "m" + text + "\x1b[0m"


BOLD, DIM = "1", "2"
RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN = "31", "32", "33", "34", "35", "36"
ORANGE = "38;5;208"


def vlen(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def pad(s: str, width: int, align: str = "<") -> str:
    gap = max(0, width - vlen(s))
    return s + " " * gap if align == "<" else " " * gap + s


def trunc(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def table(headers: List[str], rows: List[List[str]], aligns: str) -> None:
    widths = [max([vlen(h)] + [vlen(r[i]) for r in rows]) for i, h in enumerate(headers)]
    print("  " + "  ".join(c(pad(h, w, a), BOLD) for h, w, a in zip(headers, widths, aligns)))
    print("  " + c("  ".join("─" * w for w in widths), DIM))
    for r in rows:
        print("  " + "  ".join(pad(v, w, a) for v, w, a in zip(r, widths, aligns)))


def section(title: str) -> None:
    width = min(shutil.get_terminal_size((100, 20)).columns, 100)
    print()
    print(c("── " + title + " " + "─" * max(0, width - len(title) - 4), BOLD, CYAN))


def fmt_params(p: float) -> str:
    if p >= 0.95e9:
        v = p / 1e9
        return f"{v:.0f}B" if v >= 10 else f"{v:.1f}B"
    return f"{p / 1e6:.0f}M"


def fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def fmt_gb(g: float) -> str:
    return f"{g:.1f} GB" if g < 100 else f"{g:.0f} GB"


# --------------------------------------------------------------------------- #
# Detección de hardware
# --------------------------------------------------------------------------- #

@dataclass
class GPU:
    name: str
    vram_gb: float
    vendor: str            # nvidia | amd | intel | apple | other
    unified: bool = False  # memoria compartida con la CPU (Apple Silicon)
    bandwidth_gbs: float = 0.0


@dataclass
class Hardware:
    os_name: str
    arch: str
    cpu_name: str
    cores_physical: Optional[int]
    cores_logical: int
    ram_gb: float
    ram_available_gb: Optional[float]
    disk_free_gb: float
    gpus: List[GPU] = field(default_factory=list)
    cpu_bandwidth_gbs: float = 50.0

    # --- capacidades derivadas -------------------------------------------- #
    @property
    def unified_gpu(self) -> Optional[GPU]:
        return next((g for g in self.gpus if g.unified), None)

    @property
    def compute_gpus(self) -> List[GPU]:
        """GPUs dedicadas útiles para inferencia (>= 2 GB de VRAM)."""
        return [g for g in self.gpus if not g.unified and g.vram_gb >= 1.9]

    @property
    def gpu_usable_gb(self) -> float:
        u = self.unified_gpu
        if u:
            return u.vram_gb
        # ~0.6 GB por GPU reservados para el driver / escritorio
        return sum(max(0.0, g.vram_gb - 0.6) for g in self.compute_gpus)

    @property
    def gpu_bandwidth(self) -> float:
        u = self.unified_gpu
        if u:
            return u.bandwidth_gbs
        gs = self.compute_gpus
        return min(g.bandwidth_gbs for g in gs) if gs else 0.0

    @property
    def ram_usable_gb(self) -> float:
        """RAM que razonablemente puede usar un modelo sin ahogar al sistema."""
        return max(0.0, self.ram_gb - max(2.5, self.ram_gb * 0.15))


def _run(cmd: List[str], timeout: float = 8) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _os_name() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if sysname == "Windows":
        rel, build = platform.release(), platform.version()
        try:
            if rel == "10" and int(build.split(".")[-1]) >= 22000:
                rel = "11"
        except ValueError:
            pass
        return f"Windows {rel} (build {build})"
    if sysname == "Linux":
        for line in _read("/etc/os-release").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
        return f"Linux {platform.release()}"
    return f"{sysname} {platform.release()}"


def _cpu_name() -> str:
    sysname = platform.system()
    if sysname == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sysname == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif sysname == "Linux":
        for line in _read("/proc/cpuinfo").splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                return line.split(":", 1)[1].strip()
        for line in _run(["lscpu"]).splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "Desconocido"


def _physical_cores() -> Optional[int]:
    try:
        import psutil  # type: ignore
        return psutil.cpu_count(logical=False)
    except Exception:
        pass
    sysname = platform.system()
    if sysname == "Darwin":
        v = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(v) if v.isdigit() else None
    if sysname == "Linux":
        cores, phys = set(), "0"
        for line in _read("/proc/cpuinfo").splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                cores.add((phys, line.split(":")[1].strip()))
        return len(cores) or None
    return None


def _ram() -> Tuple[float, Optional[float]]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return vm.total / GIB, vm.available / GIB
    except Exception:
        pass
    sysname = platform.system()
    if sysname == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / GIB, st.ullAvailPhys / GIB
    elif sysname == "Darwin":
        total = _run(["sysctl", "-n", "hw.memsize"])
        avail = None
        vm = _run(["vm_stat"])
        m = re.search(r"page size of (\d+)", vm)
        if m:
            page = int(m.group(1))
            pages = sum(int(x) for x in re.findall(
                r"Pages (?:free|inactive|speculative|purgeable):\s+(\d+)", vm))
            avail = pages * page / GIB
        if total.isdigit():
            return int(total) / GIB, avail
    elif sysname == "Linux":
        info = {}
        for line in _read("/proc/meminfo").splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0]) * 1024 if v.split() else 0
        if "MemTotal" in info:
            return info["MemTotal"] / GIB, info.get("MemAvailable", 0) / GIB or None
    return 8.0, None  # valor conservador si todo falla


# Ancho de banda de memoria aproximado (GB/s): determina los tokens/segundo.
NVIDIA_BW = [
    ("5090", 1792), ("5080", 960), ("5070 ti", 896), ("5070", 672), ("5060 ti", 448), ("5060", 448),
    ("4090", 1008), ("4080 super", 736), ("4080", 717), ("4070 ti super", 672), ("4070 ti", 504),
    ("4070 super", 504), ("4070", 504), ("4060 ti", 288), ("4060", 272),
    ("3090", 936), ("3080 ti", 912), ("3080", 760), ("3070 ti", 608), ("3070", 448),
    ("3060 ti", 448), ("3060", 360), ("3050", 224), ("2080 ti", 616), ("2080", 448),
    ("2070", 448), ("2060", 336), ("1660", 192), ("1080 ti", 484), ("1080", 320), ("1070", 256),
    ("h200", 4800), ("h100", 3350), ("a100", 1935), ("l40", 864), ("a6000", 768), ("a5000", 768),
    ("a4000", 448), ("v100", 900), ("t4", 320),
]
AMD_BW = [
    ("9070 xt", 640), ("9070", 640), ("9060 xt", 320), ("7900 xtx", 960), ("7900 xt", 800),
    ("7900 gre", 576), ("7800 xt", 624), ("7700 xt", 432), ("7600", 288), ("6950", 576),
    ("6900", 512), ("6800", 512), ("6700", 384), ("6600", 224), ("mi300", 5300), ("mi250", 3200),
]
APPLE_BW = [  # (regex sobre el nombre del chip, GB/s)
    (r"M1 Ultra", 800), (r"M1 Max", 400), (r"M1 Pro", 200), (r"M1", 68),
    (r"M2 Ultra", 800), (r"M2 Max", 400), (r"M2 Pro", 200), (r"M2", 100),
    (r"M3 Ultra", 819), (r"M3 Max", 400), (r"M3 Pro", 150), (r"M3", 100),
    (r"M4 Max", 546), (r"M4 Pro", 273), (r"M4", 120),
    (r"M5 Max", 610), (r"M5 Pro", 307), (r"M5", 153),
]


def _lookup_bw(name: str, table_: List[Tuple[str, int]], default: float) -> float:
    n = name.lower()
    for key, bw in table_:
        if key in n:
            return float(bw)
    return default


def _vendor(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("nvidia", "geforce", "rtx", "quadro", "tesla")):
        return "nvidia"
    if any(k in n for k in ("amd", "radeon", "ati ")):
        return "amd"
    if "intel" in n or "arc" in n:
        return "intel"
    return "other"


def _gpu_from(name: str, vram_gb: float) -> GPU:
    vendor = _vendor(name)
    if vendor == "nvidia":
        bw = _lookup_bw(name, NVIDIA_BW, 300)
        if "laptop" in name.lower():
            bw *= 0.7
    elif vendor == "amd":
        bw = _lookup_bw(name, AMD_BW, 300)
    elif vendor == "intel":
        bw = 450.0 if "a770" in name.lower() or "b580" in name.lower() else 250.0
    else:
        bw = 250.0
    return GPU(name=name, vram_gb=vram_gb, vendor=vendor, bandwidth_gbs=bw)


def _nvidia_smi() -> List[GPU]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].replace(".", "").isdigit():
            gpus.append(_gpu_from(parts[0], float(parts[1]) / 1024))
    return gpus


def _windows_registry_gpus() -> List[Tuple[str, float]]:
    """Lee la VRAM real (64 bits) del registro; WMI la limita a 4 GB."""
    try:
        import winreg
    except ImportError:
        return []
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    found = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return found
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            key = winreg.OpenKey(root, sub)
            name = winreg.QueryValueEx(key, "DriverDesc")[0]
        except OSError:
            continue
        mem = 0
        for value in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
            try:
                v, _ = winreg.QueryValueEx(key, value)
                mem = int.from_bytes(v[:8], "little") if isinstance(v, bytes) else int(v)
                if mem:
                    break
            except (OSError, ValueError, TypeError):
                pass
        if mem:
            found.append((name, mem / GIB))
    return found


def _linux_amd_gpus() -> List[GPU]:
    names = [l.split(": ", 1)[-1] for l in _run(["lspci"]).splitlines()
             if re.search(r"VGA|3D|Display", l) and re.search(r"AMD|ATI", l)]
    gpus, seen = [], set()
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        real = os.path.realpath(dev)
        vram = _read(os.path.join(dev, "mem_info_vram_total")).strip()
        if real in seen or not vram.isdigit():
            continue
        seen.add(real)
        name = names[len(gpus)] if len(gpus) < len(names) else "AMD Radeon (amdgpu)"
        gpus.append(_gpu_from(name if "amd" in name.lower() else "AMD " + name, int(vram) / GIB))
    return gpus


def detect_hardware() -> Hardware:
    sysname, arch = platform.system(), platform.machine()
    ram_total, ram_avail = _ram()
    cpu = _cpu_name()
    try:
        disk = shutil.disk_usage(os.path.expanduser("~")).free / GIB
    except OSError:
        disk = 0.0

    gpus: List[GPU] = []
    cpu_bw = 50.0

    if sysname == "Darwin" and arch == "arm64":
        chip = cpu if "Apple" in cpu else "Apple Silicon"
        # macOS permite a la GPU usar ~2/3 de la RAM (≈3/4 en equipos de más de 36 GB)
        frac = 0.75 if ram_total > 36 else 0.67
        bw = next((b for pat, b in APPLE_BW if re.search(pat + r"\b", chip)), 100)
        gpus.append(GPU(name=f"{chip} (GPU integrada, memoria unificada)", vram_gb=ram_total * frac,
                        vendor="apple", unified=True, bandwidth_gbs=float(bw)))
        cpu_bw = float(bw)
    else:
        gpus.extend(_nvidia_smi())
        if sysname == "Windows":
            known = {g.name.lower() for g in gpus}
            for name, vram in _windows_registry_gpus():
                if name.lower() not in known and vram >= 1.9:
                    gpus.append(_gpu_from(name, vram))
                    known.add(name.lower())
        elif sysname == "Linux":
            gpus.extend(_linux_amd_gpus())

    return Hardware(os_name=_os_name(), arch=arch, cpu_name=cpu,
                    cores_physical=_physical_cores(), cores_logical=os.cpu_count() or 1,
                    ram_gb=ram_total, ram_available_gb=ram_avail, disk_free_gb=disk,
                    gpus=gpus, cpu_bandwidth_gbs=cpu_bw)


# --------------------------------------------------------------------------- #
# Estimación de memoria y velocidad
# --------------------------------------------------------------------------- #

# (nombre, bits efectivos por peso, descripción)
QUANTS = [
    ("FP16", 16.0, "precisión completa"),
    ("Q8_0", 8.5, "casi sin pérdida"),
    ("Q6_K", 6.6, "calidad muy alta"),
    ("Q5_K_M", 5.7, "calidad alta"),
    ("Q4_K_M", 4.85, "equilibrio recomendado"),
    ("Q3_K_M", 3.9, "pérdida notable"),
    ("Q2_K", 3.0, "pérdida alta"),
]

LEVEL_GPU, LEVEL_HYBRID, LEVEL_CPU, LEVEL_NO = 3, 2, 1, 0
LEVEL_LABEL = {
    LEVEL_GPU: ("● Excelente (GPU)", GREEN),
    LEVEL_HYBRID: ("● Aceptable (GPU+CPU)", YELLOW),
    LEVEL_CPU: ("● Lento (solo CPU)", ORANGE),
    LEVEL_NO: ("✗ No apto", RED),
}


def level_text(level: int) -> str:
    text, color = LEVEL_LABEL[level]
    return c(text, color)


def kv_cache_gb(params: float, ctx: int) -> float:
    """Estimación del KV-cache (FP16, con GQA típico) para el contexto dado."""
    return 0.18 * (params / 1e9) ** 0.6 * ctx / 4096


def weights_gb(params: float, bits: float) -> float:
    return params * bits / 8 / GIB


def required_gb(weights: float, params: float, ctx: int) -> float:
    return weights * 1.03 + kv_cache_gb(params, ctx) + 0.5  # +0.5 GB buffers del runtime


@dataclass
class Assessment:
    level: int
    required_gb: float
    tok_s: float


def assess(hw: Hardware, weights: float, params: float, active_params: float, ctx: int) -> Assessment:
    req = required_gb(weights, params, ctx)
    active_w = weights * (active_params / params if params else 1.0)
    gpu_cap, gpu_bw = hw.gpu_usable_gb, hw.gpu_bandwidth
    eff = 0.55        # eficiencia real frente al ancho de banda teórico
    overhead = 0.002  # segundos fijos por token (kernels, muestreo…)

    def speed(gpu_part: float, cpu_part: float) -> float:
        secs = (gpu_part / gpu_bw if gpu_part else 0) + cpu_part / hw.cpu_bandwidth_gbs
        return 1 / (secs / eff + overhead)

    if gpu_cap and req <= gpu_cap:
        return Assessment(LEVEL_GPU, req, speed(active_w, 0))
    if hw.unified_gpu is None and gpu_cap and req <= gpu_cap + hw.ram_usable_gb:
        frac = gpu_cap / req
        return Assessment(LEVEL_HYBRID, req, speed(active_w * frac, active_w * (1 - frac)))
    if req <= hw.ram_usable_gb:
        return Assessment(LEVEL_CPU, req, speed(0, active_w))
    return Assessment(LEVEL_NO, req, 0.0)


def best_quant(hw: Hardware, params: float, active: float, ctx: int) -> Tuple[str, Assessment]:
    """Mejor cuantización: se prefiere Q4_K_M o superior; entre ellas, el mejor nivel.
    En GPU completa gana la mayor calidad; con offload/CPU, la más ligera (más rápida).
    Q3/Q2 sólo se eligen si ninguna >= Q4 puede correr."""
    best = None
    for name, bits, _ in QUANTS:
        a = assess(hw, weights_gb(params, bits), params, active, ctx)
        score = quant_score(a, bits)
        if best is None or score > best[0]:
            best = (score, name, a)
    return best[1], best[2]


def quant_score(a: Assessment, bits: float) -> tuple:
    good = bits >= 4.5  # Q4 o superior
    if a.level == LEVEL_GPU or not good:
        # en GPU: mayor calidad (hasta Q8) mientras vaya a >= 10 tok/s
        tiebreak = (a.tok_s >= 10, bits <= 9.5, bits)
    else:
        tiebreak = (True, True, -bits)  # con offload/CPU: la más ligera es la más rápida
    return (a.level > 0 and good, a.level, tiebreak)


def max_params_for(limit_gb: float, bits: float, ctx: int) -> float:
    lo, hi = 0.0, 2e12
    for _ in range(60):
        mid = (lo + hi) / 2
        if required_gb(weights_gb(mid, bits), mid, ctx) <= limit_gb:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #

class HFError(Exception):
    pass


def hf_get(path: str, params: Optional[List[Tuple[str, str]]] = None):
    url = HF_API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": f"iacheck/{__version__}"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HFError("El modelo no existe en Hugging Face (o es privado).") from e
        if e.code in (401, 403):
            raise HFError("No encontrado: el modelo no existe, es privado o requiere HF_TOKEN "
                          "(revisa el nombre usuario/modelo).") from e
        raise HFError(f"Hugging Face respondió con error HTTP {e.code}.") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise HFError(f"No se pudo conectar con Hugging Face ({e}).") from e


# Modelos MoE cuyo nº de parámetros activos no aparece en el nombre
KNOWN_ACTIVE = {"gpt-oss-20b": 3.6e9, "gpt-oss-120b": 5.1e9, "mixtral-8x7b": 12.9e9,
                "mixtral-8x22b": 39e9, "deepseek-v3": 37e9, "deepseek-r1": 37e9, "kimi-k2": 32e9}

QUANT_REPO_RE = re.compile(r"gguf|awq|gptq|mlx|bnb|4bit|8bit|fp8|int4|int8|exl2|onnx|nvfp4|w4a16", re.I)


def params_from_name(model_id: str) -> Optional[float]:
    n = model_id.split("/")[-1].lower()
    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", n)
    if m:
        return int(m.group(1)) * float(m.group(2)) * 1e9 * 0.83
    vals = [float(v) * (1e9 if u == "b" else 1e6)
            for v, u in re.findall(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)([bm])(?![a-z0-9])", n)]
    return max(vals) if vals else None


def active_from_name(model_id: str, total: float) -> float:
    n = model_id.split("/")[-1].lower()
    for key, act in KNOWN_ACTIVE.items():
        if key in n:
            return min(act, total)
    m = re.search(r"-a(\d+(?:\.\d+)?)b", n)
    return min(float(m.group(1)) * 1e9, total) if m else total


def params_from_info(info: dict) -> Optional[float]:
    model_id = info.get("id", "")
    # En repos ya cuantizados (AWQ, GPTQ, NVFP4…) el conteo de safetensors no es fiable:
    # el nombre suele indicar el tamaño original.
    if QUANT_REPO_RE.search(model_id) and "gguf" not in model_id.lower():
        from_name = params_from_name(model_id)
        if from_name:
            return from_name
    st = info.get("safetensors") or {}
    if st.get("total"):
        return float(st["total"])
    gg = info.get("gguf") or {}
    if gg.get("total"):
        return float(gg["total"])
    return params_from_name(info.get("id", ""))


@dataclass
class ModelEval:
    id: str
    params: Optional[float]
    active: Optional[float]
    downloads: int
    likes: int
    task: str
    quant: str = ""
    assessment: Optional[Assessment] = None

    @property
    def level(self) -> int:
        return self.assessment.level if self.assessment else -1


def evaluate(info: dict, hw: Hardware, ctx: int) -> ModelEval:
    p = params_from_info(info)
    m = ModelEval(id=info["id"], params=p, active=None, downloads=info.get("downloads", 0) or 0,
                  likes=info.get("likes", 0) or 0, task=info.get("pipeline_tag") or "")
    if p:
        m.active = active_from_name(m.id, p)
        m.quant, m.assessment = best_quant(hw, p, m.active, ctx)
    return m


LIST_EXPAND = [("expand[]", x) for x in ("safetensors", "downloads", "likes", "pipeline_tag")]

# Lista de respaldo si no hay conexión (id, parámetros)
FALLBACK = [
    ("Qwen/Qwen3-0.6B", 0.75e9), ("Qwen/Qwen3-4B", 4.02e9), ("Qwen/Qwen3-8B", 8.19e9),
    ("Qwen/Qwen3-14B", 14.8e9), ("Qwen/Qwen3-32B", 32.8e9), ("Qwen/Qwen3-30B-A3B", 30.5e9),
    ("meta-llama/Llama-3.2-3B-Instruct", 3.21e9), ("meta-llama/Llama-3.1-8B-Instruct", 8.03e9),
    ("meta-llama/Llama-3.3-70B-Instruct", 70.6e9), ("google/gemma-3-4b-it", 4.3e9),
    ("google/gemma-3-12b-it", 12.2e9), ("google/gemma-3-27b-it", 27.4e9),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.25e9),
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", 24e9), ("microsoft/phi-4", 14.7e9),
    ("microsoft/Phi-4-mini-instruct", 3.84e9), ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.6e9),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 14.8e9), ("openai/gpt-oss-20b", 21e9),
    ("openai/gpt-oss-120b", 117e9), ("HuggingFaceTB/SmolLM3-3B", 3.08e9),
]


def fetch_candidates() -> Tuple[List[dict], bool]:
    """Modelos populares y en tendencia de HF. Devuelve (lista, en_linea)."""
    seen: Dict[str, dict] = {}
    try:
        for task in ("text-generation", "image-text-to-text"):
            for sort, limit in (("trendingScore", 60), ("downloads", 80)):
                if task != "text-generation" and sort == "downloads":
                    limit = 30
                data = hf_get("/models", [("pipeline_tag", task), ("sort", sort),
                                          ("limit", str(limit))] + LIST_EXPAND)
                for info in data:
                    mid = info.get("id", "")
                    if (mid and mid not in seen and not QUANT_REPO_RE.search(mid)
                            and (info.get("downloads") or 0) >= 5000):
                        seen[mid] = info
        return list(seen.values()), True
    except HFError:
        return [{"id": i, "safetensors": {"total": p}, "downloads": 0, "likes": 0,
                 "pipeline_tag": "text-generation"} for i, p in FALLBACK], False


def search_models(query: str, limit: int) -> List[dict]:
    return hf_get("/models", [("search", query), ("sort", "downloads"), ("limit", str(limit)),
                              ("expand[]", "gguf")] + LIST_EXPAND)


# --------------------------------------------------------------------------- #
# Vistas
# --------------------------------------------------------------------------- #

def print_specs(hw: Hardware) -> None:
    section("Especificaciones del equipo")
    cores = f"{hw.cores_physical} núcleos / {hw.cores_logical} hilos" if hw.cores_physical \
        else f"{hw.cores_logical} hilos"
    avail = f"  ({fmt_gb(hw.ram_available_gb)} libres ahora)" if hw.ram_available_gb else ""
    rows = [
        ("Sistema", f"{hw.os_name}  [{hw.arch}]"),
        ("CPU", f"{hw.cpu_name}  ({cores})"),
        ("RAM", f"{fmt_gb(hw.ram_gb)}{avail}"),
    ]
    if hw.gpus:
        for i, g in enumerate(hw.gpus):
            label = "GPU" if len(hw.gpus) == 1 else f"GPU {i + 1}"
            mem = f"hasta {fmt_gb(g.vram_gb)} usables por la GPU" if g.unified else f"{fmt_gb(g.vram_gb)} VRAM"
            rows.append((label, f"{g.name}  —  {mem}  (~{g.bandwidth_gbs:.0f} GB/s)"))
    else:
        rows.append(("GPU", c("No se detectó GPU dedicada compatible (se usará la CPU)", YELLOW)))
    rows.append(("Disco libre", fmt_gb(hw.disk_free_gb)))
    for k, v in rows:
        print(f"  {c(pad(k + ':', 13), BOLD)}{v}")


def capacity(hw: Hardware, ctx: int) -> List[Tuple[int, str, float, Dict[str, float]]]:
    """[(nivel, etiqueta, memoria disponible, {cuantización: parámetros máximos})]"""
    tiers = []
    if hw.gpu_usable_gb:
        tiers.append((LEVEL_GPU, "GPU completa", hw.gpu_usable_gb))
        if hw.unified_gpu is None:
            tiers.append((LEVEL_HYBRID, "GPU + RAM (offload)", hw.gpu_usable_gb + hw.ram_usable_gb))
    if hw.unified_gpu is None:
        tiers.append((LEVEL_CPU, "Solo CPU", hw.ram_usable_gb))
    out = []
    for level, label, limit in tiers:
        maxes = {q: max_params_for(limit, next(b for n, b, _ in QUANTS if n == q), ctx)
                 for q in ("Q4_K_M", "Q8_0", "FP16")}
        out.append((level, label, limit, maxes))
    return out


def print_capacity(hw: Hardware, ctx: int) -> None:
    section(f"Capacidad estimada  (contexto de {ctx} tokens)")
    notes = {LEVEL_GPU: "  (rápido)", LEVEL_HYBRID: "  (más lento)", LEVEL_CPU: "  (lento)"}
    for level, label, _, maxes in capacity(hw, ctx):
        parts = [f"~{fmt_params(p)} en {q}" if p >= 3e8 else f"— en {q}" for q, p in maxes.items()]
        print(f"  {c('●', LEVEL_LABEL[level][1])} {pad(label + ':', 22)} hasta {', '.join(parts)}"
              f"{c(notes[level], DIM)}")
    print(c("  B = miles de millones de parámetros. Q4_K_M es la cuantización más usada en "
            "Ollama / LM Studio / llama.cpp.", DIM))


def model_rows(models: List[ModelEval], numbered: bool = False) -> List[List[str]]:
    rows = []
    for i, m in enumerate(models, 1):
        name = trunc(m.id, 48)
        if m.params is None or m.assessment is None:
            row = [name, "?", fmt_num(m.downloads), "—", "—", c("? Sin datos de tamaño", DIM), ""]
        else:
            moe = c(f" (A{fmt_params(m.active)})", DIM) if m.active and m.active < m.params * 0.8 else ""
            a = m.assessment
            speed = f"~{a.tok_s:.0f} tok/s" if a.level else ""
            row = [name, fmt_params(m.params) + moe, fmt_num(m.downloads),
                   m.quant if a.level else "—", fmt_gb(a.required_gb), level_text(a.level), speed]
        rows.append(([str(i)] if numbered else []) + row)
    return rows


MODEL_HEADERS = ["Modelo", "Params", "Descargas", "Mejor quant", "Memoria", "Veredicto", "Velocidad"]


def cmd_scan(hw: Hardware, ctx: int, top: int, as_json: bool) -> None:
    candidates, online = fetch_candidates()
    evals = [evaluate(i, hw, ctx) for i in candidates]
    evals = [e for e in evals if e.params and e.params >= 1e8]
    runnable = [e for e in evals if e.level > 0]
    runnable.sort(key=lambda e: (e.level, e.params), reverse=True)

    if as_json:
        print(json.dumps({"hardware": asdict(hw), "context": ctx, "online": online,
                          "models": [{**asdict(e), "url": f"{HF_WEB}/{e.id}"} for e in runnable]},
                         indent=2, ensure_ascii=False, default=str))
        return

    print_specs(hw)
    print_capacity(hw, ctx)

    section("Modelos de Hugging Face que puedes correr" + ("" if online else " (lista sin conexión)"))
    if not online:
        print(c("  Sin conexión a Hugging Face: se muestra una lista local de modelos conocidos.", YELLOW))
    shown: List[ModelEval] = []
    for lvl in (LEVEL_GPU, LEVEL_HYBRID, LEVEL_CPU):
        group = [e for e in runnable if e.level == lvl]
        if lvl != LEVEL_GPU:  # fuera de la GPU importa más la velocidad (p. ej. modelos MoE)
            group.sort(key=lambda e: e.assessment.tok_s, reverse=True)
        if lvl == LEVEL_CPU and any(e.level == LEVEL_GPU for e in runnable):
            group = group[:5]
        shown.extend(group[: max(3, top // 3) if lvl != LEVEL_GPU else top])
    if shown:
        table(MODEL_HEADERS, model_rows(shown), "<>><><>")
    else:
        print(c("  Ningún modelo popular cabe en este equipo.", RED))
    too_big = len(evals) - len(runnable)
    print(c(f"\n  Analizados {len(evals)} modelos populares/en tendencia; {too_big} no caben en este equipo.", DIM))
    print(c("  Velocidad = estimación de generación con llama.cpp / Ollama. Busca otro modelo con: "
            "iacheck search <nombre>", DIM))


def cmd_search(hw: Hardware, ctx: int, query: str, limit: int, as_json: bool,
               interactive: bool = False) -> None:
    try:
        results = search_models(query, limit)
    except HFError as e:
        print(c(f"  {e}", RED))
        return
    evals = [evaluate(i, hw, ctx) for i in results]
    if as_json:
        print(json.dumps([{**asdict(e), "url": f"{HF_WEB}/{e.id}"} for e in evals],
                         indent=2, ensure_ascii=False, default=str))
        return
    section(f'Resultados en Hugging Face para "{query}"')
    if not evals:
        print(c("  No se encontraron modelos con ese nombre.", YELLOW))
        return
    table(["#"] + MODEL_HEADERS, model_rows(evals, numbered=True), ">" + "<>><><>")
    ok = sum(1 for e in evals if e.level >= LEVEL_HYBRID and e.assessment.tok_s >= 5)
    print(c(f"\n  {ok} de {len(evals)} resultados pueden correr localmente con fluidez (GPU o GPU+CPU).", DIM))
    if interactive:
        while True:
            choice = ask("  Número para ver el detalle (Enter para volver): ")
            if not choice:
                return
            if choice.isdigit() and 1 <= int(choice) <= len(evals):
                cmd_check(hw, ctx, evals[int(choice) - 1].id)
            else:
                print(c("  Opción no válida.", RED))
    else:
        print(c("  Detalle de un modelo: iacheck check <usuario/modelo>", DIM))


GGUF_QUANT_RE = re.compile(
    r"(UD-[A-Z0-9_]+|I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?)(?=[.\-]|$)", re.I)


@dataclass
class QuantRow:
    name: str
    desc: str
    size_gb: float          # tamaño del archivo GGUF o de los pesos estimados
    assessment: Assessment
    disk_ok: bool = True


@dataclass
class RepoAnalysis:
    ev: ModelEval
    gated: bool
    is_gguf: bool
    rows: List[QuantRow]
    chosen: Optional[QuantRow]
    available_gb: float


def normalize_repo(repo: str) -> str:
    return repo.strip().replace(HF_WEB + "/", "").replace("https://hf.co/", "").strip("/")


def fetch_repo(repo: str) -> Tuple[dict, Dict[str, float]]:
    """Descarga metadatos y, si existen, el tamaño real de los archivos GGUF."""
    repo = normalize_repo(repo)
    info = hf_get(f"/models/{repo}", [("expand[]", x) for x in
                                      ("safetensors", "gguf", "downloads", "likes",
                                       "pipeline_tag", "gated")])
    gguf_files: Dict[str, float] = {}
    try:
        for f in hf_get(f"/models/{info['id']}/tree/main", [("recursive", "1")]):
            path = f.get("path", "")
            if path.lower().endswith(".gguf") and "mmproj" not in path.lower():
                key = re.sub(r"-\d{5}-of-\d{5}(?=\.gguf$)", "", path)
                size = (f.get("lfs") or {}).get("size", f.get("size", 0))
                gguf_files[key] = gguf_files.get(key, 0) + size / GIB
    except HFError:
        pass
    return info, gguf_files


def analyze_repo(hw: Hardware, ctx: int, info: dict, gguf_files: Dict[str, float]) -> RepoAnalysis:
    ev = evaluate(info, hw, ctx)
    rows: List[QuantRow] = []
    chosen, best_score = None, None
    if ev.params:
        active = ev.active or ev.params
        if gguf_files:
            for path, size in sorted(gguf_files.items(), key=lambda kv: -kv[1]):
                m = GGUF_QUANT_RE.search(os.path.basename(path)[:-5])
                q = m.group(1) if m else os.path.basename(path)
                bits = size * GIB * 8 / ev.params
                rows.append(QuantRow(q, f"{bits:.1f} bits/peso", size,
                                     assess(hw, size, ev.params, active, ctx), size <= hw.disk_free_gb))
        else:
            for name, bits, desc in QUANTS:
                w = weights_gb(ev.params, bits)
                rows.append(QuantRow(name, desc, w, assess(hw, w, ev.params, active, ctx),
                                     w <= hw.disk_free_gb))
        for r in rows:
            score = quant_score(r.assessment, r.size_gb * GIB * 8 / ev.params)
            if best_score is None or score > best_score:
                chosen, best_score = r, score
    avail = max(hw.gpu_usable_gb + (hw.ram_usable_gb if not hw.unified_gpu else 0), hw.ram_usable_gb)
    return RepoAnalysis(ev, bool(info.get("gated")), bool(gguf_files), rows, chosen, avail)


def find_gguf_versions(model_id: str, n: int = 3) -> List[str]:
    base = model_id.split("/")[-1]
    try:
        return [i["id"] for i in hf_get("/models", [("search", f"{base} gguf"), ("sort", "downloads"),
                                                    ("limit", "15")])
                if base.lower() in i["id"].lower()][:n]
    except HFError:
        return []


def cmd_check(hw: Hardware, ctx: int, repo: str) -> None:
    try:
        info, gguf_files = fetch_repo(repo)
    except HFError as e:
        print(c(f"  {e}", RED))
        return
    ra = analyze_repo(hw, ctx, info, gguf_files)
    ev = ra.ev
    section(f"Análisis de {ev.id}")
    print(f"  {c('Enlace:', BOLD)}     {HF_WEB}/{ev.id}")
    print(f"  {c('Tarea:', BOLD)}      {ev.task or 'desconocida'}   "
          f"{c('Descargas:', BOLD)} {fmt_num(ev.downloads)}   {c('Likes:', BOLD)} {fmt_num(ev.likes)}")
    if ra.gated:
        print(c("  Acceso restringido: acepta la licencia en la web de HF antes de descargar.", YELLOW))
    if not ev.params:
        print(c("  No se pudo determinar el tamaño del modelo.", YELLOW))
        return
    moe = f"  (MoE: ~{fmt_params(ev.active)} activos por token)" if ev.active and ev.active < ev.params * 0.8 else ""
    print(f"  {c('Parámetros:', BOLD)} {fmt_params(ev.params)}{moe}")

    section("Archivos GGUF del repositorio (tamaño real)" if ra.is_gguf else "Estimación por cuantización")
    rows = [[trunc(r.name, 22), c(r.desc, DIM), fmt_gb(r.size_gb), fmt_gb(r.assessment.required_gb),
             level_text(r.assessment.level),
             f"~{r.assessment.tok_s:.0f} tok/s" if r.assessment.level else "",
             "" if r.disk_ok else c("sin espacio en disco", RED)] for r in ra.rows]
    table(["Cuantización", "Detalle", "Tamaño", "Memoria", "Veredicto", "Velocidad", ""], rows, "<<>><><")

    section("Conclusión")
    if ra.chosen and ra.chosen.assessment.level > 0:
        q, a = ra.chosen.name, ra.chosen.assessment
        print(f"  {level_text(a.level)}  —  recomendado: {c(q, BOLD)} "
              f"(~{fmt_gb(a.required_gb)} de memoria, ~{a.tok_s:.0f} tokens/s estimados)")
        if ra.is_gguf:
            print(f"  Ollama:    {c(f'ollama run hf.co/{ev.id}:{q}', CYAN)}")
            print(f"  LM Studio: busca {c(ev.id, CYAN)} y descarga el archivo {q}")
        else:
            found = find_gguf_versions(ev.id)
            if found:
                print("  Versiones GGUF listas para Ollama / LM Studio:")
                for f in found:
                    print(f"    • {c(f, CYAN)}   →  iacheck check {f}")
    else:
        need = min((r.assessment.required_gb for r in ra.rows), default=0)
        print(f"  {level_text(LEVEL_NO)} — necesita al menos {fmt_gb(need)} incluso cuantizado; "
              f"tu equipo dispone de ~{fmt_gb(ra.available_gb)}.")
        print(c("  Prueba una versión más pequeña de la misma familia.", DIM))


# --------------------------------------------------------------------------- #
# CLI e interfaz interactiva
# --------------------------------------------------------------------------- #

BANNER = r"""
  ┌──────────────────────────────────────────────────────┐
  │  iacheck  ·  ¿Qué IA local puede correr mi equipo?   │
  └──────────────────────────────────────────────────────┘"""


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def interactive(ctx: int, top: int) -> None:
    print(c(BANNER, BOLD, CYAN))
    print(c("  Detectando hardware…", DIM))
    hw = detect_hardware()
    while True:
        print()
        print(c("  ¿Qué quieres hacer?", BOLD))
        print(f"   {c('1', CYAN)}  Escanear equipo y recomendar modelos")
        print(f"   {c('2', CYAN)}  Buscar un modelo en Hugging Face")
        print(f"   {c('3', CYAN)}  Analizar un repositorio concreto (usuario/modelo)")
        print(f"   {c('4', CYAN)}  Cambiar longitud de contexto (actual: {ctx} tokens)")
        print(f"   {c('5', CYAN)}  Ver sólo especificaciones")
        print(f"   {c('0', CYAN)}  Salir")
        choice = ask(c("  > ", BOLD))
        if choice == "1":
            print(c("  Consultando Hugging Face…", DIM))
            cmd_scan(hw, ctx, top, False)
        elif choice == "2":
            q = ask("  Nombre del modelo a buscar (p. ej. llama, qwen3, gemma 3 27b): ")
            if q:
                cmd_search(hw, ctx, q, 20, False, interactive=True)
        elif choice == "3":
            r = ask("  Repositorio (p. ej. Qwen/Qwen3-8B o una URL de HF): ")
            if r:
                cmd_check(hw, ctx, r)
        elif choice == "4":
            v = ask("  Nuevo contexto en tokens (2048, 4096, 8192, 32768…): ")
            if v.isdigit() and int(v) >= 512:
                ctx = int(v)
            else:
                print(c("  Valor no válido.", RED))
        elif choice == "5":
            print_specs(hw)
            print_capacity(hw, ctx)
        elif choice in ("0", "q", "salir", "exit", ""):
            print(c("  ¡Hasta luego!", DIM))
            return
        else:
            print(c("  Opción no válida.", RED))


def main(argv: Optional[List[str]] = None) -> int:
    global USE_COLOR
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ctx", type=int, default=argparse.SUPPRESS,
                        help="longitud de contexto en tokens para la estimación (por defecto 4096)")
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS,
                        help="desactiva los colores")

    parser = argparse.ArgumentParser(
        prog="iacheck", parents=[common],
        description="Estima qué modelos de IA de Hugging Face puede correr tu equipo localmente.")
    parser.add_argument("--version", action="version", version=f"iacheck {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_scan = sub.add_parser("scan", parents=[common], help="escanea el equipo y recomienda modelos")
    p_scan.add_argument("--top", type=int, default=15, help="modelos a mostrar en GPU (por defecto 15)")
    p_scan.add_argument("--json", action="store_true", help="salida en JSON")

    p_search = sub.add_parser("search", parents=[common], help="busca modelos y dice si son aptos")
    p_search.add_argument("query", nargs="+", help="texto a buscar")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--json", action="store_true", help="salida en JSON")

    p_check = sub.add_parser("check", parents=[common], help="analiza un repositorio concreto de HF")
    p_check.add_argument("repo", help="usuario/modelo o URL de Hugging Face")

    sub.add_parser("specs", parents=[common], help="muestra sólo las especificaciones del equipo")
    sub.add_parser("tui", parents=[common], help="abre la interfaz de terminal (por defecto)")
    sub.add_parser("menu", parents=[common], help="menú de texto simple (sin interfaz a pantalla)")

    args = parser.parse_args(argv)
    if getattr(args, "no_color", False) or getattr(args, "json", False):
        USE_COLOR = False
    ctx = max(512, getattr(args, "ctx", 4096))

    try:
        if args.cmd is None:
            args.cmd = "tui" if sys.stdin.isatty() and sys.stdout.isatty() else "scan"
            args.top, args.json = 15, False
        if args.cmd == "tui":
            try:
                import iacheck_tui
            except ImportError:
                print(c("  La interfaz necesita la librería 'textual':  pip install textual", YELLOW))
                print(c("  Mientras tanto se abre el menú de texto.\n", DIM))
                interactive(ctx, 15)
                return 0
            iacheck_tui.run(ctx)
            return 0
        if args.cmd == "menu":
            interactive(ctx, 15)
            return 0
        hw = detect_hardware()
        if args.cmd == "scan":
            cmd_scan(hw, ctx, args.top, args.json)
        elif args.cmd == "search":
            cmd_search(hw, ctx, " ".join(args.query), args.limit, args.json)
        elif args.cmd == "check":
            cmd_check(hw, ctx, args.repo)
        elif args.cmd == "specs":
            print_specs(hw)
            print_capacity(hw, ctx)
        print()
    except KeyboardInterrupt:
        print()
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
