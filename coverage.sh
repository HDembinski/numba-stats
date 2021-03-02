#!/bin/sh
poetry run pytest --cov=numba_stats --cov-report=html
