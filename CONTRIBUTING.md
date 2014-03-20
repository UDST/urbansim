Contributing to UrbanSim
========================

Style
-----

- Python code should follow the [PEP 8 Style Guide][pep8].
- Python docstrings should follow the [NumPy documentation format][numpydoc].

### Imports

Imports should be one per line.
Imports should be grouped into standard library, third-party,
and intra-library imports. `from` import should follow "regular" `imports`.
Within each group the imports should be alphabetized.
Here's an example:

```python
import sys
from glob import glob

import numpy as np

import urbansim.urbansim.modelcompile as modelcompile
from urbansim.util import misc
```

Imports of scientific Python libraries should follow these conventions:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
```


Thanks!

[pep8]: http://legacy.python.org/dev/peps/pep-0008/
[numpydoc]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
