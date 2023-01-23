# Trainable RNA Structural Aligner
# Installation
This project is written mainly in Rust, a systems programming language.
You need to install Rust components, i.e., rustc (the Rust compiler), cargo (the Rust package manager), and the Rust standard library.
Visit [the Rust website](https://www.rust-lang.org) to see more about Rust.
You can install Rust components with the following one line:
```bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
The above installation is done by [Rustup](https://github.com/rust-lang-nursery/rustup.rs), and Rustup enables to easily switch a compiler in use.
You can install ConsAlign as follows: 
```bash
$ # AVX, SSE, and MMX enabled for rustc (another example: RUSTFLAGS='--emit asm -C target-feature=+avx2 -C target-feature=+ssse3 -C target-feature=+mmx -C target-feature=+fma')
$ RUSTFLAGS='--emit asm -C target-feature=+avx -C target-feature=+ssse3 -C target-feature=+mmx' cargo install consalign
```
Check if you have installed ConsAlign properly as follows:
```bash
$ consalign # Its available command options will be displayed.
```

# Train ConsAlign Using Your Training Data
To train ConsAlign, you need to install ConsTrain, a training tool of ConsProb (ConsAlign's back engine):
```bash
$ git clone https://github.com/heartsh/consprob-trained && cd consprob-trained
$ RUSTFLAGS='--emit asm -C target-feature=+avx -C target-feature=+ssse3 -C target-feature=+mmx' cargo install --path . -f
```
You can pass ConsTrain your training data as follows:
```bash
$ cd scripts
$ # Your trained parameters will appear at "../src/trained_feature_score_sets.rs".
$ constrain -i train_data_dir_path -o train_log_file_path
```
Now, you can install ConsAlign with your trained parameters:
```bash
$ cd ../..
$ git clone https://github.com/heartsh/consalign && cd consalign
$ # Before executing the below command, replace 'consprob-trained = "X.Y"' with 'consprob-trained = {path = "../consprob-trained"}' in "./Cargo.toml" to designate your trained parameters
$ RUSTFLAGS='--emit asm -C target-feature=+avx -C target-feature=+ssse3 -C target-feature=+mmx' cargo install --path . -f
$ # ConsAlign parameterized with your trained parameters will be called
$ consalign
```

# Docker Playground <img src="./assets/images_fixed/docker_logo.png" width="40">
I offer [my Docker-based playground for RNA software and its instruction](https://github.com/heartsh/rna-playground) to replay my computational experiments easily.

# Method Digest
[ConsProb-Turner](https://github.com/heartsh/consprob) and [ConsProb-Trained](https://github.com/heartsh/consprob-trained) infer sparse posterior matching/base-pairing probabilities on RNA pairwise structural alignment using [Turner's model and the CONTRAfold models](https://github.com/heartsh/rna-ss-params).
This repository offers ConsAlign, transfer-learned RNA structural aligner combining ConsProb-Turner and ConsProb-trained.
When you run ConsAlign, it automatically determines its alignment prediction hyper-parameters by maximizing expected sum-of-pairs scores.

# Author
[Heartsh](https://github.com/heartsh)

# License
Copyright (c) 2018 Heartsh  
Licensed under [the MIT license](http://opensource.org/licenses/MIT).
