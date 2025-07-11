Installation
============

The following instructions will allow you to install PyRAY.

Prerequisites
-------------

PyRAY uses the Python modules included in the list below. This module
officially supports Python 3.9+.

1. lmfit
2. numpy
3. PyIRI


Installation Options
--------------------

1. Clone the git repository
::


   git clone https://github.com/victoriyaforsythe/PyRAY.git


2. Install PyRAY:
   Change directories into the repository folder and build the project.
   There are a few ways you can do this:

   A. Install on the system (root privileges required)::


        sudo pip install .

   B. Install at the user level::


        pip install --user .

   C. Install with the intent to change the code::


        pip install --user -e .

Optional Requirements
---------------------

To run the test suite and build the documentation locally, you also need the
following Python packages.

.. extras-require:: test
    :pyproject:

.. extras-require:: doc
    :pyproject:
