# MLEncrypt-Research

## Usage

Replace the `single` command with `hparams` if you want to run hyperparameter optimization.

Note: XLA isn't actually used as of now, since decorating `iterate` with `@tf.function(experimental_compile=True)` returns many errors.

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
-   <https://github.com/drsimonj/blogR/blob/master/Rmd/pretty-scatter-plots-of-correlated-vars.Rmd>
    -   used for creating 2-dimensional scatterplots
