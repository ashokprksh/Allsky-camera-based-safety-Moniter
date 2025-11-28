# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['allsky_monitor_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('allsky_cloud_detector_final.tflite', '.'),  # Model file
        ('labels.txt', '.'),                          # Labels file
        ('allsky_monitor_config.json', '.'),          # Configuration file
    ],
    hiddenimports=[
          'cv2', 'paramiko', 'tensorflow', 'tensorflow.lite.python.interpreter', 
        'numpy.core._dtype_ctypes', 'PIL.Image', 'PIL.ImageTk', 'json'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AllSkyMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
