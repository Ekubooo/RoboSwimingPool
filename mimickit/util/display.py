import atexit
import os
import subprocess
import time

from util.logger import Logger

_display_process = None
_display_name = None

def ensure_virtual_display(display: str = ":99") -> None:
    """Start Xvfb virtual display if no DISPLAY is set. Needed for headless Vulkan rendering.
    
    If DISPLAY is already set, uses it (assumes it's valid). Otherwise starts Xvfb on the
    specified display number.
    """
    global _display_process
    global _display_name
    if "DISPLAY" in os.environ:
        return
    
    try:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        Logger.print("WARNING: Xvfb not found. Install with: apt-get install xvfb")
        Logger.print("Headless camera rendering may not work without a virtual display.")
        return
    _display_process = process
    _display_name = display
    atexit.register(close_virtual_display)
    time.sleep(1)
    if process.poll() is not None:
        close_virtual_display()
        Logger.print("WARNING: Xvfb exited before the virtual display was ready.")
        return
    os.environ["DISPLAY"] = display
    Logger.print("Started virtual display on {}".format(display))


def close_virtual_display():
    """Stop our Xvfb child only; leave externally supplied displays untouched."""
    global _display_process
    global _display_name
    process = _display_process
    display = _display_name
    _display_process = None
    _display_name = None
    atexit.unregister(close_virtual_display)
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()
    if os.environ.get("DISPLAY") == display:
        os.environ.pop("DISPLAY", None)
