# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compatibility entry point for the fixed 80/12/9 migration subset."""

from __future__ import annotations

import sys

from scripts.libero_three.run_combined_batch import main

if __name__ == "__main__":
    raise SystemExit(main(["--profile", "migration_subset", *sys.argv[1:]]))
