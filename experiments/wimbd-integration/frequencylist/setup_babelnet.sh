#!/bin/bash

conda create -y -n babelnet python=3.8.13
conda activate babelnet
pip install babelnet[rpc]