#!/usr/bin/env bash

set -o nounset
set -o pipefail
set -o errexit

. ~/.bashrc
conda activate timeseries-api

exec uvicorn main:app --reload