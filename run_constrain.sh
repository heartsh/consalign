#! /usr/bin/env sh

TRAIN_DATA="$1"
shift
TRAIN_LOG="$1"
shift

RUSTFLAGS='--emit asm -C target-feature=+avx -C target-feature=+ssse3 -C target-feature=+mmx' cargo install consprob-trained
cd consprob-trained/scripts
constrain -i $TRAIN_DATA -o $TRAIN_LOG
sed -E -i "s/^(consprob-trained=)\"[0-9]+\.[0-9]+\"$/\1={path = \"\.\/consprob-trained\"}/" ./Cargo.toml
cd ../..
RUSTFLAGS='--emit asm -C target-feature=+avx -C target-feature=+ssse3 -C target-feature=+mmx' cargo install --path . -f
