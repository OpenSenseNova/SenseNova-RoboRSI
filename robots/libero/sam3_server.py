# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Launch the image's SAM3 server without duplicating its implementation."""

import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ.get("ROBORSI_IMAGE_TOOLS_ROOT", "/opt/roborsi-image-tools"))
    if not root.exists() and "ROBORSI_IMAGE_TOOLS_ROOT" not in os.environ:
        root = Path("/opt/roborsi")
    name = "roborsi/robots/components/sam3_server.py"
    lock = json.loads(Path(__file__).with_name("image_tools.lock.json").read_text())
    if (
        hashlib.sha256((root / name).read_bytes()).hexdigest()
        != lock["service_sources"][name]
    ):
        raise ValueError("Image SAM3 server differs from pinned version")
    sys.path.insert(0, str(root))
    from roborsi.robots.components.sam3_server import main as serve

    return serve()


if __name__ == "__main__":
    main()
