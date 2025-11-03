# escape=`

FROM mcr.microsoft.com/windows/server:ltsc2022
LABEL maintainer="sunny.titus@deltares.nl"

ARG PIXI_VERSION=v0.59.0

## Setup user
USER "NT Authority\System"

# Install .NET 4.8
ADD https://download.visualstudio.microsoft.com/download/pr/2d6bb6b2-226a-4baa-bdec-798822606ff1/8494001c276a4b96804cde7829c04d7f/ndp48-x86-x64-allos-enu.exe /ndp48-x86-x64-allos-enu.exe
RUN C:\ndp48-x86-x64-allos-enu.exe /quiet /install && del C:\ndp48-x86-x64-allos-enu.exe

## Install chocolatey
ENV ChocolateyUseWindowsCompression false 
RUN powershell Set-ExecutionPolicy Bypass -Scope Process -Force;`
     [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;`
     iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

## Install useful packages
RUN choco install -y --no-progress `
    git.install `
    powershell-core

## Install Pixi
RUN ["powershell", "iwr -useb https://pixi.sh/install.ps1 | iex"]

CMD [ "cmd" ]