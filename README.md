# Machine Learning Frameworks Research

Exploration of Machine Learning concepts and popular Python frameworks to explore supervised learnings, basic models, and neural networks.

## Description

## Module: Python and TensorFlow - Tony Manschula, Henry Shires, and Alex Bashara

(CPR E 487 at Iowa State University): Our deliverables will consist of a modified Lab 1 Jupyter notebook using PyTorch. Additionally, we will also benchmark against TensorFlow in several key areas in a writeup presentation for the final demo:

- Usability
  - Ease of use of the API
  - Clarity and usefulness of API documentation
  - Any lab 1 activities that could not be implemented in PyTorch
- Training accuracy
  - Train a given model using a different number of epochs and report the number that achieved the highest validation accuracy
- Training time/performance
  - What hardware acceleration options do the frameworks support?
  - How much training time did each framework require to achieve its best validation accuracy?
- Resource utilization
  - Memory utilization and how each framework may lend itself to a given selection of hardware (embedded, etc.)

Read entire project report: [487_report.pdf](./docs/487_report.pdf)

## Module: Scikit-Learn - Henry Shires

My playground using the Scikit-Learn machine learning library for Python, following Google's *Machine Learning Recipes* by Josh Gordon. I desired to develop a fundamental understanding of machine learning and how it implements into source code to solve problems. I then used the knowledge I gained to learn about how Unity enables developers to utilize ML in 3D game design. I hoped to implement their tools into a game of my own, however I was unsuccessful.

## Installation

### Create Python Virtual Environment

Guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

1. `python -m venv venv`
2. `.\venv\Scripts\activate` for Windows or `source venv/bin/activate` for Unix
3. `python -m pip install --upgrade pip`
4. `pip install -r ./pytorch/requirements.txt` or `./tensorflow/requirements.txt` or `./scikit-learn/requirements.txt`

### VS Code Extensions

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [PyTorch Snippets](https://marketplace.visualstudio.com/items?itemName=SBSnippets.pytorch-snippets)

## Resources

### PyTorch Tutorial

Follow instructions from here: https://pytorch.org/tutorials/beginner/basics/intro.html

Use the [./pytorch/tutorial](./pytorch/tutorial) directory of notebooks to get started:

1. Intro
2. Quickstart
3. Tensors
4. Datasets & DataLoaders
5. Transforms
6. Build Model
7. Autograd
8. Optimization
9. Save & Load Model

### Scikit-Learn Recipes

- [Scikit-Learn Documentation](https://scikit-learn.org)
- [Google Developers ML Recipes](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)
- [Neural Networks Playground](https://playground.tensorflow.org)
- [Unity 3D ML Agents](https://www.youtube.com/watch?v=32wtJZ3yRfw&index=2&list=PLX2vGYjWbI0R08eWQkO7nQkGiicHAX7IX&t=0s)

### PyTorch Equivalent of TF Profiler and Tensorboard

[Using the PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality)
This generates a trace.json file that can be imported into various viewers, the most prominent being [Perfetto UI.](https://ui.perfetto.dev/)

Addtitionally, we can directly print inference times in the program using a [built-in PyTorch method.](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-execution-time)
