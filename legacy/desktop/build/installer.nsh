; Image Trace NSIS Custom Installer Script
; This file is included by electron-builder during Windows NSIS packaging.

!macro customInstall
  ; No custom install steps needed — VC++ Redist is handled
  ; by the user's system or pre-installed on modern Windows 10+.
!macroend

!macro customUnInstall
  ; No custom uninstall steps needed
!macroend
