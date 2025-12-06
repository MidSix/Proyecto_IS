import sys
import os
import ctypes
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon


def get_icon_path(is_win: bool) -> str:
    """Get the platform-specific icon file path.

    Returns the appropriate icon path based on the operating system:
    Windows uses .ico format, other systems use .jpg format.

    Parameters
    ----------
    is_win : bool
        True if running on Windows, False otherwise.

    Returns
    -------
    str
        Full path to the icon file for the current platform.
    """
    base_path = os.path.dirname(__file__)
    if is_win:
        return os.path.join(base_path, "..", "..", "assets", "icons",
                                                                "app.ico")
    else:
        return os.path.join(base_path, "..", "..", "assets", "icons",
                                                                "app.jpg")

def set_app_icon(app: QApplication) -> None:
    """Set the application window icon.

    Configures the application icon for the current platform. On
    Windows, also sets a unique app model ID for taskbar integration.

    Parameters
    ----------
    app : QApplication
        The PyQt5 application instance to set the icon for.

    Returns
    -------
    None
    """
    is_sys_win = sys.platform.startswith("win")
    if is_sys_win:
        myappid = "com.chad.linearregressionapp"  # ID único
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        icon = get_icon_path(is_sys_win)
    else:
        icon = get_icon_path(is_sys_win)
    app.setWindowIcon(QIcon(icon))
    return None