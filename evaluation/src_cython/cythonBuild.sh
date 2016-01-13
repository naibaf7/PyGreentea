#!/usr/bin/env bash
echo "STARTING BUILD"
python setup.py build_ext --inplace
printf "\nFINISHED BUILD\n"