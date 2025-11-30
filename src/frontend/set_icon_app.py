import sys
import os
import ctypes
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon


def get_icon_path(is_win: bool) -> str:
    base_path = os.path.dirname(__file__)
    if is_win:
        return os.path.join(base_path, "..", "assets", "icons", "app.ico")
    else:
        return os.path.join(base_path, "..", "assets", "icons", "app.jpg")

def set_app_icon(app: QApplication) -> None:
    is_sys_win = sys.platform.startswith("win")
    if is_sys_win:
        myappid = "com.chad.linearregressionapp"  # ID único
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        icon = get_icon_path(is_sys_win)
    else:
        icon = get_icon_path(is_sys_win)
    app.setWindowIcon(QIcon(icon))
    return None