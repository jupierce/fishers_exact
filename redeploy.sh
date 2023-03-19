#!/bin/bash

docker build -t quay.io/jupierce/fishers:prod -f Dockerfile .
docker push quay.io/jupierce/fishers:prod

echo You must restart the compute instance!