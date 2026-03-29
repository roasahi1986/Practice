#!/bin/bash

#----------------------------- working directory -----------------------------#
REAL_FILE=$(readlink -f $0)
SCRIPT_NAME=${REAL_FILE##*/}
SCRIPT_DIR=$(cd "$(dirname "${REAL_FILE}")"; pwd)

pushd ${SCRIPT_DIR}
source ${SCRIPT_DIR}/script/functional.sh
source ${SCRIPT_DIR}/script/deps.sh

#------------------------------ trap debug info ------------------------------#
trap 'error_info ${SCRIPT_NAME} ${LINENO} $?' ERR

#----------------------------------- build -----------------------------------#
setup_deps
setup
clean
check
build_tools
perf_demo_graph
