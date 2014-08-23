.. Install page 

============
Installation
============

There are two options for installing *sick*: you can either install it with ``pip`` (recommended), or download the source and install it yourself.


Install with ``pip``
------------------

With ``sudo`` access
^^^^^^^^^^^^^^^^^^^^

The easiest way to install *sick* is by using the ``pip`` (or ``easy_install``, `if you must <https://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install>`_) installer:

``sudo pip install sick``

And that's it. You're done!

Without ``sudo`` access
^^^^^^^^^^^^^^^^^^^^^^^

If you don't have ``sudo`` access then use the ``--user`` flag with ``pip`` like this:

``pip install sick --user``

If the ``--user`` flag is used then you'll have to check that the folder containing the ``sick`` command line function is on your ``$PATH``. The folder location will change depending on your system architecture. ``pip`` will announce the folder during the installation, and ``~/.local/bin/`` is probably a good place to start. 


Download the Source
-------------------
The source code can be downloaded directly from the GitHub repository `here <https://github.com/andycasey/sick/archive/master.zip>`_. Unpack the archive, then to install the software use:

``python setup.py install``

The installer will make the ``sick`` Python module available on your path, and install a command-line function called ``sick``. 

If the installer requires ``sudo`` access and you don't have it, you can use:

``python setup.py install --user``

If you install with the ``--user`` flag, then you may need to check *where* the installer has placed the ``sick`` script and ensure that folder is on your ``$PATH``. The location will be in the log file (displayed on screen) and will probably be something like ``~/.local/bin`` 

