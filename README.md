Based on renderer from "Two-shot SVBRDF capture for stationary materials" by Aittala, Miika and Weyrich, Tim and Lehtinen, Jaakko. https://mediatech.aalto.fi/publications/graphics/TwoShotSVBRDF/

# Dependencies
## Core
 * Python 3.5.2
 * Vispy 0.5.0 or above
 * GLFW 3
 * SciPy
 * Numpy

# Setup
First install dependencies and their dependencies.

## Virtualenv
Setup virtualenv.
```
pyenv install 3.5.2
pyenv virtualenv --system-site-packages 3.5.2 myenv
pyenv activate
```

## Install Python Packages
```
pip install git+https://github.com/vispy/vispy.git@master
pip install -r requirements.txt
```
