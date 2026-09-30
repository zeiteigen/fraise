#!/usr/bin/env bash
# Install Intel's user-space NPU libraries without changing the system driver.
set -euo pipefail

npu_runtime_path="${HOME}/.local/share/fraise/npu-runtime"
npu_staging_path=$(mktemp -d)
trap 'rm -rf "$npu_staging_path"' EXIT

command -v bsdtar >/dev/null || { echo 'Install libarchive (bsdtar) first.' >&2; exit 1; }
curl --fail --location --output "$npu_staging_path/driver.tar.gz" \
  'https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz'
curl --fail --location --output "$npu_staging_path/loader.deb" \
  'https://github.com/oneapi-src/level-zero/releases/download/v1.34.0/libze1_1.34.0%2Bu24.04_amd64.deb'
printf '%s  %s\n' \
  '1efcd4b60c22abee751d8f2705962cbcc2a569de45c7e0e670cf08afbfcdc1d2' "$npu_staging_path/driver.tar.gz" \
  '45210e4549cd965ad7b9f160eefe53cbcdba01af341bea7ac7ba0aedeeb613ba' "$npu_staging_path/loader.deb" \
  | sha256sum --check

bsdtar -xf "$npu_staging_path/driver.tar.gz" -C "$npu_staging_path"
mkdir "$npu_staging_path/extracted"
for npu_package in \
  "$npu_staging_path"/intel-level-zero-npu_*.deb \
  "$npu_staging_path"/intel-driver-compiler-npu_*.deb \
  "$npu_staging_path/loader.deb"; do
  bsdtar -xOf "$npu_package" 'data.tar.*' | bsdtar -xf - -C "$npu_staging_path/extracted"
done
mkdir -p "$npu_runtime_path/lib"
cp -a "$npu_staging_path/extracted/usr/lib/x86_64-linux-gnu/." "$npu_runtime_path/lib/"
printf 'Installed Intel NPU driver 1.38.0 and Level Zero loader 1.34.0 in %s\n' "$npu_runtime_path"
