#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Matthew Priester
# Builds libmadshift_kernels.so (the fast trellis search loaded by madshift/kernels.py) with nvcc.
# GPU architecture: CUDA_ARCH (for example sm_89, sm_90, sm_120); default: the GPU of this machine (native).
set -e
cd "$(dirname "$0")"
nvcc -O3 -shared -Xcompiler -fPIC -o libmadshift_kernels.so kernels.cu -arch="${CUDA_ARCH:-native}"
