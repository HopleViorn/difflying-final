# Difflying
Diffsim project for drone planning and navigation

## Installation

Install Warp from Source, [link](https://github.com/NUS-HcRL/warp) here.

To install the flyinglib package, run the following commands in the root
directory of this repository:

```bash
# install flyinglib requirements
pip install -r requirements.txt
# install flyinglib
pip install -e .
```

## Demo
Run Command:
```bash
python examples/train_free.py
python examples/test_free.py
```

## TODO

1. Align with the drone from T-lab (**Yanrui**)
   1. Geometry description
   2. Physical parameters
2. Collision-free exp (**Shuyi**)
   1. Add collision scenes
   2. Design reward functions
   3. Modify policy backbone and train
