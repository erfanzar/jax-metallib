# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Support for JAX / jaxlib 0.10.x

### Changed
- Bumped Python dependency pins to `jax>=0.10.0,<0.11` and `jaxlib>=0.10.0,<0.11`
- Updated native dependency pins to the XLA `b6f37ab` / StableHLO `3a8886d` / LLVM `8918319` triple matching jaxlib 0.10.0 bytecode
- Bumped `_BUILT_FOR_JAXLIB` runtime version sentinel to `(0, 10)`

### Fixed
-

### Removed
-

## [0.9.4.10] - 2024

### Added
- Initial jax-metallib PJRT plugin for Silicon and M Chips ([01469a0](https://github.com/example/jax-silicon/commit/01469a0))
- Native Metal kernels for elementwise, reduce, GEMM, LayerNorm, gather/scatter operations ([4120e2f](https://github.com/example/jax-silicon/commit/4120e2f))

### Fixed
- Hardened control-flow lowering and zero-size tensor handling in pjrt_plugin ([38e63d3](https://github.com/example/jax-silicon/commit/38e63d3))

[Unreleased]: https://github.com/example/jax-silicon/compare/v0.9.4.10...HEAD
[0.9.4.10]: https://github.com/example/jax-silicon/releases/tag/v0.9.4.10
