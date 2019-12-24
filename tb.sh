#!/usr/bin/env bash
if [ $1 == "tj" ]; then
  ssh tj "./tb.sh snowy"
elif [ $1 == "snowy" ]; then
  cd MLEncrypt-Research
  pipenv shell
  tensorboard --logdir logs/hparams/
else
  echo "Invalid argument"
  exit 128
fi
