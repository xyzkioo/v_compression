"""Identity fixed to this independent source project."""
import os
import sys
from pathlib import Path
COMPONENT = 'model_compression'
APPS = {COMPONENT: {"slug": 'v_compression', "title": 'v_compression', "tool": 'compression'}}
def configure(component=None):
    if component is not None and component != COMPONENT:
        raise ValueError("此项目只能启动自身组件")
    os.environ["VTOOLS_COMPONENT"] = COMPONENT
    folder = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    os.environ.setdefault("VTOOLS_USER_DATA", str(folder / "user-data"))
def identity():
    configure()
    return {"component": COMPONENT, "title": 'v_compression', "primary_tool": 'compression', "portable": True, "updates_enabled": False}
