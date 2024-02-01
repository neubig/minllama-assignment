#!/usr/bin/env bash

conda create -n llama_hw python=3.11
conda activate llama_hw

conda install pytorch==2.0.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch
pip install tqdm==4.66.1
pip install requests==2.31.0
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==1.2.2
pip install numpy==1.26.3
pip install tokenizers==0.13.3
wget https://www.cs.cmu.edu/~vijayv/stories42M.pt