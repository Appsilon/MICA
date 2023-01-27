import numpy as np

from datasets.creation.generator import Generator
from datasets.creation.instances.example import Example

np.random.seed(42)

if __name__ == '__main__':

    generator = Generator([Example()])

    generator.run()
