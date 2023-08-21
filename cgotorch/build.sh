#!/bin/bash

set -x

BASE_DIR=$(realpath $(dirname $0))
BUILD_DIR="$BASE_DIR/build"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

cmake ..
make $1
cp -f *.so "$BASE_DIR"
cp -f *.dylib "$BASE_DIR"
rm -rf "$BUILD_DIR"
