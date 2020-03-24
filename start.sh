#!/usr/bin/env bash

gunicorn api:app -k uvicorn.workers.UvicornWorker
