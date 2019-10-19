Installation notes
==================

shtns
-----

Download and decompress the sources, go in the root directory and
run something like::

  ./configure --prefix=$HOME/.local --enable-openmp --enable-python
  make

Then::
  
  make install

or to install just for the user::

  python setup.py install --user
  
or, activate a virtual environment and::

  python setup.py install



  
