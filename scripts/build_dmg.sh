#!/bin/bash
set -e

# Script expects path to the extracted .app
APP_PATH=$1

if [ -z "$APP_PATH" ]; then
    echo "Usage: $0 /path/to/SwiftBuddy.app"
    exit 1
fi

APP_NAME=$(basename "$APP_PATH")

echo "=========================================="
echo "1. Zip the App for Notarization"
echo "=========================================="
ZIP_PATH="/tmp/${APP_NAME}.zip"
/usr/bin/ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

echo "=========================================="
echo "2. Submit to Apple Notary Service"
echo "=========================================="
# Submit and wait for the result
xcrun notarytool submit "$ZIP_PATH" \
    --apple-id "$APPLEID_USERNAME" \
    --password "$APPLEID_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" \
    --wait

echo "=========================================="
echo "3. Staple the Notarization Ticket"
echo "=========================================="
# Staple it so it can pass Gatekeeper even offline
xcrun stapler staple "$APP_PATH"

echo "=========================================="
echo "4. Package into DMG"
echo "=========================================="
mkdir -p output
DMG_NAME="SwiftBuddy-macOS.dmg"

create-dmg \
  --volname "SwiftBuddy" \
  --volicon "$APP_PATH/Contents/Resources/AppIcon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "SwiftBuddy.app" 200 190 \
  --hide-extension "SwiftBuddy.app" \
  --app-drop-link 600 185 \
  "output/$DMG_NAME" \
  "$APP_PATH"

echo "=========================================="
echo "SUCCESS! Created notarized output/$DMG_NAME"
echo "=========================================="
