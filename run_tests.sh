#!/bin/bash

. .venv/bin/activate
# DEV=1 to enable DEV mode, which will generate random keys for testing, since
# KMS is not available in non-TEE test environment.
PYTHONPATH=src DEV=1 python -m pytest tests/ -v "$@"
