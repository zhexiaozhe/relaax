#!/usr/bin/env bash
python ../../relaax/server/parameter_server/main.py --config config.yaml --bind localhost:7000 --checkpoint-dir checkpoints/boxing_a3c --log-level WARNING