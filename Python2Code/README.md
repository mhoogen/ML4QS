# ML4QS

This repository provides all the code associated with the book titled "Machine Learning for the Quantified Self", authored by Mark Hoogendoorn and Burkhardt Funk and published by Springer in 2017. The website of the book can be found on ml4qs.org

The Python code requires the following packages to run everything (note that we indicate the version we have tested with):
- Python version 2.7.12
- Anaconda version 4.2.0
    - matplotlib version 1.5.3
    - statsmodels version 0.6.1
    - scikit-learn version 0.18.1
    - numpy 1.11.1
    - pandas 0.19.2
    - scipy version 0.18.1
    - nltk version 3.2.1 
- pyflux version 0.4.14
- pykalman version 0.9.5
- gensim version 0.13.3
- pyclust version 0.1.15
- inspyred version 1.0.1
- pybrain version 0.3

Create and activate an anaconda environment and install all package versions using `conda install --name <EnvironmentName> --file conda_requirements.txt`.
Install non-conda packages using pip: `pip install -r pip_requirements.txt`.

Note that we have tried to make the code as robust as we can, but we cannot provide any guarantees on its correctness. This code is made available under the GNU public license. We have used snippets of code from other sources and have tried to add references to these in our code where possible. When using the code for publications, please include a reference to the book in your paper:

Hoogendoorn, M. and Funk, B., Machine Learning for the Quantified Self - On the Art of Learning from Sensory Data, Springer, 2017.

