#!/bin/bash

cd /site/LOL_analiz


setup_venv() {
    echo "Setting up virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
}

if [ ! -d ".venv" ]; then
    setup_venv
else
    source .venv/bin/activate
fi

cd bin

npm run dev &

echo "Starting Python API..."
uvicorn main:app --host 0.0.0.0 --port 8008 &

wait