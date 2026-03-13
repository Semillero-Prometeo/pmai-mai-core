"""CLI entry point – ``python -m pmai_core``."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def main() -> None:
    settings_path = Path("settings.toml")

    # Allow overriding settings path via CLI argument
    if len(sys.argv) > 1:
        settings_path = Path(sys.argv[1])

    from pmai_core.settings import Settings

    settings = Settings.from_toml(settings_path)

    from pmai_core.app import run_app

    asyncio.run(run_app(settings))


if __name__ == "__main__":
    main()
