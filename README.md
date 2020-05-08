# MLEncrypt-Research

## Usage

Note: Replace the `single` command with `hparams` if you want to run hyperparameter optimization.

Note: It may appear that enabling JIT compilation results in a slowdown. If you don't want to run with JIT, then don't pass the XLA flags to the run command.

Note: Our script features very verbose logging to TensorBoard. Enable this by passing `-tb` to the run command.

Note: TensorBoard logging does not currently work with XLA. Comment out the image summaries for weights if you want to use XLA with TensorBoard logging.

### CPU

```zsh
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" poetry run python cli.py single
```

### GPU

```zsh
TF_XLA_FLAGS=--tf_xla_auto_jit=2 poetry run python cli.py single
```

## Acknowledgements

-   <https://github.com/farizrahman4u/neuralkey>
    -   Mr. Fariz Rahman's implementation was the initial basis for ours
-   <https://github.com/brigan/NeuralCryptography>
    -   ported Dr. Luís F. Seoane's C++ implementation of the probabilistic attack to TensorFlow
