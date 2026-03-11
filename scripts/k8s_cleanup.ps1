$ErrorActionPreference = "Stop"

kubectl delete -k k8s/base --ignore-not-found=true
