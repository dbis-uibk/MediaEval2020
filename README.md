# MediaEval2020: Emotions and Themes in Music

This repository contains the implementation of our, team UIBK-DBIS, solution to the
[Emotions and Themes in Music](https://multimediaeval.github.io/editions/2020/tasks/music/)
task as part of [MediaEval 2020](https://multimediaeval.github.io/editions/2020/).

The used tag groups can be found in the respective run configurations that can
be found in the `plans` folder.

## Repository Structure

The `results` folder contains the predictions, decisions and results of our
five submitted runs. The according source code can be found in the `src`
folder and is under BSD-2-Clause license.

## Usage/Reproduction

To run commands in the `Makefile` be sure that you run them using `pipenv`. The
easiest way to do so is to execute them within `pipenv shell`.

After linking the dataset to the appropriate location of the `data` folder running
`pipenv run dbispipeline-link` it is possible to reproduces the dataset using the
scripts in the `tools` folder.

Finally, the results can be reproduced by executing the provided plans in the
`plans` folder with the following call:
`pipenv run python -m dbispipeline plans/<plan-file.py>`

## Contributing

Please use the [pre-commit](https://pre-commit.com/) hooks. Either install it
on your system or use the development dependencies.
