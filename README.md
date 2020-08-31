# MediaEval2020

MediaEval2020 contains DBIS pipeline configurations.


## Requirements

This project requires dataset doi://... to be accessible in `data/raw`.


## Usage/Reproduction
To run commands in the `Makefile` be sure that you them using `pipenv`. The
easiest way to do so is to execute them within `pipenv shell`.

1. preprocess the data by calling ..., after which it should be in `data/processed`.
2. call `pipenv run python -m dbispipeline <yourconfigurationfile.py>`

## Contributing
Please use the [pre-commit](https://pre-commit.com/) hooks. Either install it
on your system or use the development dependencies.
