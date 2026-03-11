#!/usr/bin/env bash
set -euo pipefail

kubectl delete -k k8s/base --ignore-not-found=true
