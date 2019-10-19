Simple .emacs.d setup directory for Python and Latex
====================================================

Python is wonderful but being a dynamic language, a good editor with
flycheck is essential. Emacs is a very good generalist editor which
can be quite good for Python with a correct setup. Unfortunately emacs
is rather difficult to setup so many people do not use at all its
capacities and write very bad Python code with it.

Here we provide a tool to easily setup emacs to make it an acceptable
Python editor.

Dependencies
------------

- emacs > 24

- cask (https://github.com/cask/cask, management tool for Emacs)

Installation
------------

Copy-paste these lines in a terminal::

  hg clone https://bitbucket.org/fluiddyn/fluid_emacs.d && \
    mv ~/.emacs.d ~/previous_.emacs.d

  mv fluid_emacs.d ~/.emacs.d && cd ~/.emacs.d && cask

To uninstall::

  mv ~/.emacs.d ~/fluid_emacs.d
  mv ~/previous_.emacs.d ~/.emacs.d
  rm -rf ~/fluid_emacs.d
