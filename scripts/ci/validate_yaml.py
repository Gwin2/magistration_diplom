"""Parse repository YAML files and fail with file-level diagnostics."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SKIP_PARTS = {".git", ".venv", "venv", "workspace_state"}


def yaml_files() -> list[Path]:
    paths = {path for pattern in ("**/*.yml", "**/*.yaml") for path in ROOT.glob(pattern)}
    return sorted(path for path in paths if not SKIP_PARTS.intersection(path.parts))


def main() -> int:
    files = yaml_files()
    errors: list[str] = []
    for path in files:
        try:
            list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        except (OSError, yaml.YAMLError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")

    if errors:
        print("YAML validation failed:")
        print("\n".join(errors))
        return 1

    print(f"YAML validation passed: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
