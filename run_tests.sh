#!/bin/bash

. .venv/bin/activate
PYTHONPATH=src DEV=1 python -m pytest tests/ -v "$@"
