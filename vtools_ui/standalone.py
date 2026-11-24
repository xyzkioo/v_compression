"""Identity fixed to this independent source project."""
import os
import sys
from pathlib import Path
COMPONENT = 'model_compression'
APPS = {COMPONENT: {"slug": 'v_compression', "title": 'v_compression', "tool": 'compression'}}


def is_system_install() -> bool:
    return getattr(sys, "frozen", False) and (Path(sys.executable).parent / ".system-install").is_file()


def configure(component=None):
    if component is not None and component != COMPONENT:
        raise ValueError("此项目只能启动自身组件")
    os.environ["VTOOLS_COMPONENT"] = COMPONENT
    folder = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    data_dir = folder / "user-data"
    if is_system_install():
        xdg_data = Path(os.environ.get("XDG_DATA_HOME", ""))
        data_dir = (xdg_data if xdg_data.is_absolute() else Path.home() / ".local" / "share") / "v_compression"
    os.environ.setdefault("VTOOLS_USER_DATA", str(data_dir))
def identity():
    configure()
    return {"component": COMPONENT, "title": 'v_compression', "primary_tool": 'compression', "portable": True, "updates_enabled": False}
