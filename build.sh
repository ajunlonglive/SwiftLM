#!/bin/bash
set -e

echo "=> Initializing submodules..."
git submodule update --init --recursive

echo "=> Building SwiftLM (release)..."
swift build -c release

echo "=> Copying default.metallib..."
# Dynamically find the default.metallib in case of path changes in submodules
METALLIB_SRC=$(find . -name "default.metallib" | grep -v "\.build" | head -n 1)
METALLIB_DEST=".build/arm64-apple-macosx/release/"

# Also resolving the generic release symlink folder just in case
METALLIB_DEST_SYMLINK=".build/release/"

if [ -n "$METALLIB_SRC" ] && [ -f "$METALLIB_SRC" ]; then
    mkdir -p "$METALLIB_DEST"
    cp "$METALLIB_SRC" "$METALLIB_DEST"
    
    if [ -d "$METALLIB_DEST_SYMLINK" ]; then
        cp "$METALLIB_SRC" "$METALLIB_DEST_SYMLINK"
    fi
    
    # Also copying to root to be safe
    cp "$METALLIB_SRC" ./
    
    echo "✅ Successfully copied default.metallib from $METALLIB_SRC"
else
    echo "⚠️  Warning: default.metallib not found anywhere in the project! MLX GPU operations may fail."
fi

echo "=> Build complete!"
