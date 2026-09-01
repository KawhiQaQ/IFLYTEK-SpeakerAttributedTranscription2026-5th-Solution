#!/usr/bin/env bash

if [ "${XUNFEI_S2_OLD_LD_LIBRARY_PATH+x}" = x ]; then
    export LD_LIBRARY_PATH="${XUNFEI_S2_OLD_LD_LIBRARY_PATH}"
    unset XUNFEI_S2_OLD_LD_LIBRARY_PATH
fi
