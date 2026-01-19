from setuptools import setup
from pathlib import Path

APP = ['yt_view_gui_mac_pro.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'app.icns' if Path('app.icns').exists() else None,
    'packages': [
        'PySide6', 'numpy', 'pandas', 'sklearn',
        'xgboost', 'shap', 'matplotlib', 'joblib', 'scipy', 'PIL'
    ],
    'includes': [
        'shap.explainers._tree',
        'PIL.Image', 'PIL._imaging', 'imp'
    ],
    'excludes': [
        'PyInstaller', 'PyQt5', 'PyQt6', 'pytest', 'tkinter',
        'xgboost.testing', 'PySide6.scripts.deploy_lib', 'project_lib',
        'wheel', 'pip', 'setuptools', 'pkg_resources'
    ],
    'plist': {
        'CFBundleName': 'YouTube View Predictor',
        'CFBundleDisplayName': 'YouTube View Predictor',
        'CFBundleIdentifier': 'com.paveil.ytview',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': True
    },
    'frameworks': [
        "/opt/homebrew/opt/libffi/lib/libffi.8.dylib"
    ]
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app']
)
