#!/bin/bash

set -x
set -e

# Detect OS
OS=$(uname)
BASE_DIR=$(realpath $(dirname $0))
BUILD_DIR="$BASE_DIR/build"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

cmake ..
make $1

if [ "$OS" == "Darwin" ]; then
    # macOS
    cp -f *.dylib "$BASE_DIR"
else
    cp -f *.so "$BASE_DIR"
    exit 1
fi

rm -rf "$BUILD_DIR"
