#!/bin/bash
# Start the edit ranking app (macOS: double-click; Linux: bash run_mac.command).
# The first run installs what the app needs into .venv (a few minutes, needs internet).
cd "$(dirname "$0")" || exit 1

if [ ! -f .venv/.installed ]; then
  echo "First run: installing the app. This takes a few minutes..."
  rm -rf .venv
  if ! python3 -m venv .venv; then
    echo "Python 3.10 or newer is needed: https://www.python.org/downloads/"
    read -r -p "Press Enter to close."
    exit 1
  fi
  if ! .venv/bin/python -m pip install -r requirements.txt; then
    echo "Installing failed. Check the internet connection and run this file again."
    read -r -p "Press Enter to close."
    exit 1
  fi
  touch .venv/.installed
fi

echo "Starting the app. It opens in your browser; keep this window open while you rank."
.venv/bin/python app.py
