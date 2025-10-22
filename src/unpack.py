import ctypes
import os
import shutil
import sys
import tarfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
dataset_dir = os.path.join(ASSETS_DIR, "INRIAPerson")
tar_file = os.path.join(ASSETS_DIR, "INRIAPerson.tar")


def unpack_dataset():
    """使用python解压tar文件(包含符号链接)"""
    shutil.rmtree(dataset_dir, ignore_errors=True)
    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(path=ASSETS_DIR)


def run_as_admin():
    """
    检查当前是否为管理员权限，如果不是，则请求UAC弹窗重新运行自身。
    返回 True 表示当前已是管理员身份。
    """
    try:
        # 判断是否为管理员
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        is_admin = False

    if not is_admin:
        # 重新启动自身，并请求管理员权限（会弹UAC）
        params = " ".join([f'"{arg}"' for arg in sys.argv])
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, params, None, 1
        )
        sys.exit(0)


if __name__ == "__main__":
    run_as_admin()
    unpack_dataset()
