# MLEncrypt-Research

## Usage

-   `single`: runs a single instance of neural key exchange
-   `multiple`: runs many instances of neural key exchange, all with the same configuration (this can be useful for benchmarking)
-   `hparams`: runs hyperparameter optimization (different instances of neural key exchange will likely have different configurations)

Note: It may appear that enabling JIT compilation results in a slowdown. If you don't want to run with JIT, then don't pass the XLA flags to the run command.

Note: Our script features very verbose logging to TensorBoard. Enable this by passing `-tb` to the run command.

Note: You can choose to not calculate the synchronization score for each iteration. Enable this by passing `-b` to the run command.

### CPU

```zsh
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" poetry run mlencrypt-research single
```

### GPU

```zsh
TF_XLA_FLAGS=--tf_xla_auto_jit=2 poetry run mlencrypt-research single
```

## Acknowledgements

-   <https://github.com/farizrahman4u/neuralkey>
    -   Mr. Fariz Rahman's implementation was the initial basis for ours
-   <https://github.com/brigan/NeuralCryptography>
    -   ported Dr. Luís F. Seoane's C++ implementation of the probabilistic attack to TensorFlow
