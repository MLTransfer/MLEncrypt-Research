#!/usr/bin/env bash
if [ $1 == "tj" ]; then
  ssh -L 6006:127.0.0.1:6007 tj "cd MLEncrypt-Research; lsof -ti:6007 | xargs kill -9; ./tb.sh snowy"
elif [ $1 == "snowy" ]; then
  ssh -L 6007:127.0.0.1:6006 snowy "cd MLEncrypt-Research; /home/2022sratna/.local/bin/pipenv run tensorboard --logdir logs/hparams/ --port 6006"
else
  echo "Invalid argument"
  exit 128
fi
