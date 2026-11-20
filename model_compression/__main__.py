"""Stable component command-line interface."""

from vtools_runtime.component import component_main


def main() -> int:
    return component_main("model_compression")


if __name__ == "__main__":
    raise SystemExit(main())
