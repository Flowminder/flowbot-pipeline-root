#! /bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`/flowbot_dataclasses
pipenv run cookiecutter "$@"