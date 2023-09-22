# Models

The `generate` module for CoCoCroLa contains the code for generating images in folders that are structured
as the CoCoCroLa metrics expect. While it is possible to write your own runscripts to load the CoCoCroLa word
lists and generate the images yourself using any model, writing an `ImageGenerator` wrapper for your model makes
your life easier, allowing you to directly use our CLI runscripts, and allows you to contribute your model to the
benchmark itself!

## How to add a model

Adding a model to the `generate` module is pretty straightforward! You will make edits in the following places:

1. Implement the `ImageGenerator` class for your model and save it in a new python script in the models folder
2. Come up with a code for your model, and add it to the `SUPPORTED_MODULES` list in `__init__.py`
3. Add the imports and constructor for your model as an option in `get_generator` in `__init__.py`
4. (Optional, if you want to contribute your model to the package) Add the requirements for using your model as an extras flag with options in `extras_require` in `setup.py`

## Writing an ImageGenerator

See the `DiffusersImageGenerator` and `OpenAIImageGenerator` classes in `huggingface_diffusers.py` and `openai.py` respectively for examples of fully-featured and partially-featured `ImageGenerator` classes.

Full guide might be written later. For now, if you have any questions, feel free to write an issue.

## Writing patches for classes used in your ImageGenerator

If you want to add some functionality for an experiment (eg, the mid-pipeline seed swapping functionality that we added to the `diffusers` models), write those modified/patched classes in files inside `patches/`