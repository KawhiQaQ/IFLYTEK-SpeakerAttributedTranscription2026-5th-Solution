#!/usr/bin/env bash

if [[ "${CAST_OLD_LD_LIBRARY_PATH+x}" == x ]]; then
  export LD_LIBRARY_PATH="${CAST_OLD_LD_LIBRARY_PATH}"
  unset CAST_OLD_LD_LIBRARY_PATH
fi
