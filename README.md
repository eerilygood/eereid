# What is eereid?
eereid is a minimalist Python library, allowing users to train and evaluate re-identification models with ease.

Built on tensorflow and developed as part of a research project at TU Dortmund University, the library aims to be a light-weight alternative to commonly used libraries for deep learning and re-identification in particular. The library allows you to easily use your own datasets and model. In doing so, you can tweak training parameters and evaluate your models in a breeze, even if you are not particularly familiar with deep learning.

# How can I install it?

On your preferred OS, make sure you are using python 3.8 or 3.9: https://www.python.org/downloads/release/python-380/

Open up your preferred directory and clone this repository:

```git clone https://github.com/psorus/eereid.git```

Navigate to the installation directory and run install.sh in a terminal to install the required modules

```sh install.sh```

And that's it - With only five required modules installed, you are now good to go!

# How can I get started?

To see if the installation went well, you can navigate to the tests folder and run main.py. This will train and test a simple CNN on MNIST and you will be provided with Ranked Accuracy and mAP for evaluation purposes.
In main.py, you can now tinker with training and testing parameters such as:
* Loss functions
* Distance functions
* Model selection
* Dataset selection
* Data preprocessing
* Training duration & early stopping
* Training folds
* Dataset split
* Novelty detection

# How can I use my own datasets?

Navigate to the tests folder and open create_datasets.py. Depending on the labeling of your data, you can add the path to your dataset here and simply import it by running create_datasets.py, which will generate an npz-file out of your dataset.

# How can I use my own models?

TBD

# What about all those other files and folders?

* tests:
* eereid:
  * datasets
  * distances
  * experiments
  * losses
  * models
  * modifier
  * novelty
  * prepros
