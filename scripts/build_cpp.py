"""Cross-platform C++17 build script for Intransitive native core and tablebase generator."""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
CPP_DIR = ROOT_DIR / "cpp"
NATIVE_DIR = ROOT_DIR / "intransitive" / "native"
BIN_DIR = ROOT_DIR / "bin"


def find_gxx() -> str:
    gxx = shutil.which("g++")
    if gxx:
        return gxx
    msys_gxx = Path(r"C:\msys64\ucrt64\bin\g++.exe")
    if msys_gxx.exists():
        return str(msys_gxx)
    raise RuntimeError("Could not find g++ in PATH or C:\\msys64\\ucrt64\\bin\\g++.exe")


def run_cmd(cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, env=env)


def build_target(target: str, portable: bool = False) -> None:
    gxx = find_gxx()
    is_windows = platform.system() == "Windows"
    gxx_dir = str(Path(gxx).parent)
    extra_env = {"PATH": gxx_dir + os.pathsep + os.environ.get("PATH", "")} if is_windows else None

    common_flags = ["-O3", "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-function"]
    if not portable:
        common_flags.append("-march=native")

    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    if target in ("all", "tb_gen"):
        tb_src = CPP_DIR / "tb_gen.cpp"
        if not tb_src.exists():
            raise FileNotFoundError(f"Missing {tb_src}")
        tb_out = BIN_DIR / ("tb_gen.exe" if is_windows else "tb_gen")
        cmd = [gxx, *common_flags, "-pthread", str(tb_src), "-o", str(tb_out)]
        if is_windows:
            cmd.extend(["-static", "-static-libgcc", "-static-libstdc++"])
        run_cmd(cmd, extra_env=extra_env)
        print(f"Built {tb_out}")

    if target in ("all", "core"):
        api_src = CPP_DIR / "c_api.cpp"
        if not api_src.exists():
            raise FileNotFoundError(f"Missing {api_src}")
        lib_out = NATIVE_DIR / ("libintransitive_core.dll" if is_windows else "libintransitive_core.so")
        cmd = [gxx, *common_flags, "-shared"]
        if not is_windows:
            cmd.extend(["-fPIC", "-pthread"])
        else:
            cmd.append("-pthread")
        cmd.extend([str(api_src), "-o", str(lib_out)])
        if is_windows:
            cmd.extend(["-static", "-static-libgcc", "-static-libstdc++"])
        run_cmd(cmd, extra_env=extra_env)
        print(f"Built {lib_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Intransitive C++17 native binaries.")
    parser.add_argument("--target", choices=["all", "core", "tb_gen"], default="all")
    parser.add_argument("--portable", action="store_true", help="Omit -march=native for portable Colab/cloud builds")
    args = parser.parse_args()
    build_target(args.target, portable=args.portable)


if __name__ == "__main__":
    main()
