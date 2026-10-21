!macro NSIS_HOOK_POSTINSTALL
  DetailPrint "Configuring Python virtual environment..."
  DetailPrint "Running setup-venv.bat..."
  nsExec::ExecToLog '"$SYSDIR\cmd.exe" /c ""$INSTDIR\setup-venv.bat" --unattended"'
  Pop $0
  ${If} $0 != 0
    DetailPrint "Python virtual environment setup returned exit code: $0"
  ${Else}
    DetailPrint "Python virtual environment setup completed successfully."
  ${EndIf}
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  RMDir /r "$INSTDIR\.venv"
!macroend
