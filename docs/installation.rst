Installation Guide
==================

.. note::
   Since ``zoom`` involves the processing and integration of multiple modalities of data, a relatively 
   large number of Python packages are involved. Installation via ``pip`` will only automatically include 
   those packages essential for the core workflow of ``zoom``. Nevertheless, during development, we also 
   prepared several functions for the preprocessing of spatial brain phenotypes (SBPs) and single-cell RNA 
   sequencing (scRNA-seq) dataset, which necessitate the configuration of additional Python packages or 
   the setup of an R environment.

Installation requirements
-------------------------

``zoom`` was developed and tested using Python 3.9. Backwards‑compatibility and forwards‑compatibility with 
other Python 3 versions are expected but not guaranteed. All Python package version dependencies are handled 
automatically if installing ``zoom`` via ``pip`` (as described below).

- `Python 3 <https://www.python.org/>`_

Python packages for data processing:

- `numpy <http://numpy.org/>`_
- `pandas <https://pandas.pydata.org/>`_
- `nibabel <http://nipy.org/nibabel>`_
- `scanpy <https://scanpy.readthedocs.io/en/stable/>`_
- `sklearn <https://scikit-learn.org/stable/>`_
- `scipy <https://scipy.org/>`_
- `statsmodels <https://www.statsmodels.org/stable/index.html>`_

Python packages for parallel computation:

- `joblib <https://joblib.readthedocs.io/en/stable/>`_
- `tqdm <https://tqdm.github.io/>`_

Dependencies
------------

In addition to the Python packages listed above, we also prepared several functions for the preprocessing 
of spatial brain phenotypes (SBPs) and single-cell RNA sequencing (scRNA-seq) dataset. If you hope to prepare 
anatomically comprehensive Allen Human Brain Atlas (AHBA) gene expression profile optimized for cortical 
samples, please install `abagen <https://abagen.readthedocs.io/en/stable/index.html>`_. Additionally, 
if you hope to preprocess SBPs and implement spatial permutation test, please also install 
`neuromaps <https://netneurolab.github.io/neuromaps/index.html>`_. We also provided python interface for 
running `hdWCNA <https://smorabit.github.io/hdWGCNA/>`_, if needed, please set up appropriate R environment 
following the guidance of `hdWCNA tutorial <https://smorabit.github.io/hdWGCNA/>`_.

Installation
------------


``zoom`` can be downloaded from `GitHub repository <https://github.com/SpaTrek/ZOOM>`_:

.. code-block:: bash

    $ conda create -n zoom_env python=3.9
    $ conda activate zoom_env
    $ cd ZOOM-main

Install other requirements:

.. code-block:: bash

    $ pip install -r requirements.txt

Install ``zoom``:

.. code-block:: bash

    $ pip install .
