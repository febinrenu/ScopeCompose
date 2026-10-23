@echo off
chcp 65001 >nul
setlocal EnableExtensions
title ScopeCompose
rem ==========================================================================
rem  ScopeCompose - one-click launcher for Windows.
rem
rem    start.bat          build the site and open it   (http://localhost:4173, and on your network)
rem    start.bat dev      live-reloading dev server    (http://localhost:5173, and on your network)
rem    start.bat verify   check the browser engines against the recorded Python
rem
rem  Needs only Node.js 20.19 or newer. Python is NOT needed to run the site.
rem  Set SC_NO_OPEN=1 to stop it opening a browser window.
rem
rem  Phones and tablets: the server is also reachable on your local network,
rem  and the links are printed when it starts. The phone must be on the same
rem  Wi-Fi. Set SC_LOCAL_ONLY=1 to keep it on this computer only (for
rem  example on public Wi-Fi).
rem ==========================================================================

set "ROOT=%~dp0"
set "SITE=%ROOT%site"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=run"

if not exist "%SITE%\package.json" (
  echo [x] Could not find "%SITE%\package.json".
  echo     Keep start.bat in the repository root, next to the "site" folder.
  goto :fail
)

rem ---- 1. Node.js ---------------------------------------------------------
where node >nul 2>nul
if errorlevel 1 goto :nonode

node -e "const [a,b]=process.versions.node.split('.').map(Number);process.exit(a>20||(a===20&&b>=19)?0:1)"
if errorlevel 1 goto :oldnode
for /f "delims=" %%v in ('node -v') do echo [ok] Node.js %%v

cd /d "%SITE%" || goto :fail

rem ---- 2. Dependencies (first run only, or after package-lock changes) ------
if not exist "node_modules\vite\bin\vite.js" goto :install
rem Reinstall if package-lock.json is newer than the installed tree.
node -e "const f=require('fs');try{process.exit(f.statSync('package-lock.json').mtimeMs>f.statSync('node_modules/.package-lock.json').mtimeMs+1000?1:0)}catch{process.exit(1)}"
if errorlevel 1 goto :install
echo [ok] Dependencies already installed
goto :deps_done

:install
echo [..] Installing dependencies (first run takes a minute)...
call npm ci --no-audit --no-fund
if errorlevel 1 (
  echo [!] npm ci failed - retrying with npm install...
  call npm install --no-audit --no-fund
  if errorlevel 1 (
    echo [x] Could not install dependencies. Check your internet connection and try again.
    goto :fail
  )
)
echo [ok] Dependencies installed
:deps_done

set "OPEN=--open"
if "%SC_NO_OPEN%"=="1" set "OPEN="

rem Listen on the local network too, so a phone on the same Wi-Fi can open it.
set "HOST=0.0.0.0"
if "%SC_LOCAL_ONLY%"=="1" set "HOST=localhost"

if /i "%MODE%"=="dev" goto :dev
if /i "%MODE%"=="verify" goto :verify
if /i "%MODE%"=="run" goto :run
echo [x] Unknown option "%MODE%". Use: start.bat  ^|  start.bat dev  ^|  start.bat verify
goto :fail

rem ---- 3a. Build and serve the production site -------------------------------
:run
echo [..] Checking the browser engines against the recorded Python answers...
node tools\verify_engines.js >nul
if errorlevel 1 (
  echo [!] The engine check reported a disagreement. Run "start.bat verify" for details.
  echo     Continuing so the site can still be viewed.
) else (
  echo [ok] Engines agree with the Python implementation
)
echo [..] Building the site...
call node node_modules\vite\bin\vite.js build --logLevel error
if errorlevel 1 (
  echo [x] The build failed - see the messages above.
  goto :fail
)
echo [ok] Built to site\dist
echo.
call :port 4173 || goto :fail
call :links %PORT%
call node node_modules\vite\bin\vite.js preview --port %PORT% --strictPort --host %HOST% %OPEN%
goto :end

rem ---- 3b. Dev server with hot reload ------------------------------------------
:dev
call :port 5173 || goto :fail
call :links %PORT%
call node node_modules\vite\bin\vite.js --port %PORT% --strictPort --host %HOST% %OPEN%
goto :end

rem ---- 3c. Full engine verification -----------------------------------------------
:verify
node tools\verify_engines.js
if errorlevel 1 goto :fail
goto :end

rem ---- No Node.js -------------------------------------------------------------------
:nonode
echo [x] Node.js is not installed (or not on PATH).
where winget >nul 2>nul
if errorlevel 1 goto :nonode_manual
echo.
choice /c YN /m "Install Node.js LTS now with winget"
if errorlevel 2 goto :nonode_manual
winget install --id OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
if errorlevel 1 goto :nonode_manual
echo.
echo [ok] Node.js installed. CLOSE this window and run start.bat again
echo      (Windows only picks up the new PATH in a fresh window).
goto :end
:nonode_manual
echo     Install the LTS version from https://nodejs.org , then run start.bat again.
goto :fail

:oldnode
for /f "delims=" %%v in ('node -v') do echo [x] Node.js %%v is too old. ScopeCompose needs 20.19 or newer.
echo     Install the current LTS from https://nodejs.org and run start.bat again.
goto :fail

rem ---- A free port, so the links printed are the ones that work -------------------
:port
set "PORT="
for /f "delims=" %%p in ('node tools\free_port.mjs %1 %HOST%') do set "PORT=%%p"
if "%PORT%"=="" (
  echo [x] Could not find a free port near %1. Close other servers and try again.
  exit /b 1
)
exit /b 0

rem ---- Where to open it: this computer, and phones on the same network ------------
:links
echo.
echo  ===================================================================
echo   ScopeCompose
echo.
echo   On this computer:   http://localhost:%1
if "%HOST%"=="localhost" goto :links_local
echo.
echo   On your phone or tablet (same Wi-Fi as this computer):
node "%ROOT%site\tools\lan_links.mjs" %1
echo.
echo   If the phone cannot connect:
echo     - Windows may ask whether Node.js may use the network: choose Allow.
echo       On college, hotel or cafe Wi-Fi tick "Public networks" too, because
echo       Windows usually files those networks as Public.
echo     - Check that the phone is on the same Wi-Fi, not on mobile data.
echo     - Some office or college networks stop devices seeing each other;
echo       a phone hotspot that this computer joins works instead.
:links_local
echo.
echo   Keep this window open while you use the site. Ctrl+C stops it.
echo  ===================================================================
echo.
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1

:end
endlocal
exit /b 0
