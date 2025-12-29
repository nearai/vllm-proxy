#!/bin/bash

. .venv/bin/activate
# DEV=1 to enable DEV mode, which will generate random keys for testing
PYTHONPATH=src DEV=1 python -m pytest tests/ -v "$@"
