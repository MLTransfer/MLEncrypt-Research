#!/usr/bin/env bash
if [ $1 == "tj" ]; then
  ssh -L 6006:127.0.0.1:6007 tj "./tb.sh snowy"
elif [ $1 == "snowy" ]; then
  ssh -L 6007:127.0.0.1:16006 snowy "cd MLEncrypt-Research; pipenv shell; tensorboard --logdir logs/hparams/"
else
  echo "Invalid argument"
  exit 128
fi
