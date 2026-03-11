# Contributing to altor-vec

Thanks for your interest in contributing! Here's how to get started.

## Building from source

```bash
git clone https://github.com/altor-lab/altor-vec.git
cd altor-vec
cargo build
```

## Running tests

```bash
cargo test                     # core tests
cargo test --all-features      # include serialization tests
```

## Building WASM

```bash
cargo install wasm-pack        # one-time setup
cd wasm && wasm-pack build --target web --release
```

## Code style

We use standard Rust tooling — please run these before submitting a PR:

```bash
cargo fmt                      # format code
cargo clippy --all-targets --all-features -- -D warnings  # lint
```

## Pull request process

1. Open an issue describing the change you'd like to make.
2. Fork the repo and create a feature branch from `main`.
3. Make your changes, add tests, and ensure `cargo test --all-features` passes.
4. Run `cargo fmt` and `cargo clippy` with no warnings.
5. Submit a PR — we'll review it as soon as we can.

## Questions?

Open an issue or email [anshul@altorlab.dev](mailto:anshul@altorlab.dev).
