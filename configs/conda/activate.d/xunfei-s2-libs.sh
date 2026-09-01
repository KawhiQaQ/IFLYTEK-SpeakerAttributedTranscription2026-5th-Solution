#!/usr/bin/env bash

# Prefer the environment's C++/FFmpeg libraries over Ubuntu's system copies.
export XUNFEI_S2_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
