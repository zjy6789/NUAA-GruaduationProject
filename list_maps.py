"""
List available StarCraft II / SMAC map names for quick checking.

Usage:
  python list_maps.py
"""

from pathlib import Path
import os
import re


def list_smac_maps(project_root: Path):
    """Read map names from HARL SMAC map registry without importing heavy deps."""
    smac_maps_file = project_root / "HARL" / "harl" / "envs" / "smac" / "smac_maps.py"

    if not smac_maps_file.exists():
        print("\n[SMAC registry maps in HARL] file not found")
        return

    text = smac_maps_file.read_text(encoding="utf-8")
    names = sorted(set(re.findall(r'"([^"]+)"\s*:\s*\{', text)))

    print("\n[SMAC registry maps in HARL]")
    for n in names:
        print(f"  - {n}")


def list_tuned_config_maps(project_root: Path):
    """List map names that already have tuned configs."""
    smac_cfg_root = project_root / "HARL" / "tuned_configs" / "smac"
    smacv2_cfg_root = project_root / "HARL" / "tuned_configs" / "smacv2"

    if smac_cfg_root.exists():
        maps = sorted([p.name for p in smac_cfg_root.iterdir() if p.is_dir()])
        print("\n[Maps with tuned configs: SMAC]")
        for n in maps:
            print(f"  - {n}")

    if smacv2_cfg_root.exists():
        maps = sorted([p.name for p in smacv2_cfg_root.iterdir() if p.is_dir()])
        print("\n[Maps with tuned configs: SMACv2]")
        for n in maps:
            print(f"  - {n}")


def list_local_sc2_map_files():
    """Search common local folders for *.SC2Map and print map names."""
    candidates = [
        Path.home() / "Documents" / "StarCraft II" / "Maps",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "StarCraft II" / "Maps",
        Path(os.environ.get("PROGRAMFILES", "")) / "StarCraft II" / "Maps",
    ]

    found = []
    for root in candidates:
        if root.exists():
            for p in root.rglob("*.SC2Map"):
                found.append(p)

    print("\n[Local .SC2Map files in common folders]")
    if not found:
        print("  - None found in default locations.")
        print("    If your maps are elsewhere, please move/symlink to StarCraft II/Maps.")
        return

    for p in sorted(found):
        print(f"  - {p.stem}    ({p})")


def main():
    project_root = Path(__file__).resolve().parent
    print("Map inspector for this project")
    list_smac_maps(project_root)
    list_tuned_config_maps(project_root)
    list_local_sc2_map_files()


if __name__ == "__main__":
    main()
