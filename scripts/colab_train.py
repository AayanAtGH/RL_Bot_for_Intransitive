"""Turnkey Google Colab training runner with Drive persistence, C++ build, and 4-piece TB setup."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab T4/L4 AlphaZero v2 Training Runner")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/intransitive_az")
    parser.add_argument("--duration-minutes", type=float, default=None)
    parser.add_argument("--total-games", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--parallel-games", type=int, default=128)
    parser.add_argument("--sp-batch", type=int, default=256)
    parser.add_argument("--full-sims", type=int, default=320)
    parser.add_argument("--fast-sims", type=int, default=64)
    parser.add_argument("--eval-every-games", type=int, default=2000)
    parser.add_argument("--build-tb-if-missing", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_dir)
    drive_ckpt_dir = drive_root / "checkpoints"
    drive_tb_dir = drive_root / "tablebase"
    drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
    drive_tb_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build portable C++17 native engine + tb_gen
    run([sys.executable, "scripts/build_cpp.py", "--target", "all", "--portable"])

    # 2. Sync 4-piece tablebase from Drive or generate it directly to local SSD
    local_tb_dir = ROOT_DIR / "data" / "tablebase"
    local_tb_dir.mkdir(parents=True, exist_ok=True)
    drive_wdls = list(drive_tb_dir.glob("*.wdl"))
    local_wdls = list(local_tb_dir.glob("*.wdl"))

    if len(local_wdls) < 47 and len(drive_wdls) >= 47:
        print(f"Copying 4-piece tablebase ({len(drive_wdls)} slices) from Google Drive to local SSD...")
        for f in drive_tb_dir.iterdir():
            if f.suffix in (".wdl", ".dtz"):
                shutil.copy2(f, local_tb_dir / f.name)
    elif len(local_wdls) < 47 and args.build_tb_if_missing:
        print("Generating 2-4 piece tablebase on Colab CPU...")
        run([str(ROOT_DIR / "bin" / "tb_gen"), "--max-pieces", "4", "--out", str(local_tb_dir)])
        print("Backing up generated tablebase to Google Drive...")
        for f in local_tb_dir.iterdir():
            if f.suffix in (".wdl", ".dtz"):
                shutil.copy2(f, drive_tb_dir / f.name)

    # 3. Launch batched FP16 AlphaZero training saving directly to Google Drive
    train_cmd = [
        sys.executable,
        "-m",
        "intransitive.rl.trainer",
        "--blocks", str(args.blocks),
        "--channels", str(args.channels),
        "--parallel-games", str(args.parallel_games),
        "--sp-batch", str(args.sp_batch),
        "--full-sims", str(args.full_sims),
        "--fast-sims", str(args.fast_sims),
        "--eval-every-games", str(args.eval_every_games),
        "--tb-dir", str(local_tb_dir),
        "--checkpoint", str(drive_ckpt_dir / "alphazero_v2_latest.pt"),
        "--best-checkpoint", str(drive_ckpt_dir / "alphazero_v2_best.pt"),
        "--buffer-path", str(drive_ckpt_dir / "replay_buffer.npz"),
    ]
    if args.duration_minutes is not None:
        train_cmd.extend(["--duration-minutes", str(args.duration_minutes)])
    if args.total_games is not None:
        train_cmd.extend(["--total-games", str(args.total_games)])
    if args.duration_minutes is None and args.total_games is None:
        train_cmd.append("--continuous")

    run(train_cmd)


if __name__ == "__main__":
    main()
