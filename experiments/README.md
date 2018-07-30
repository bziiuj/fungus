# Technical

In each script we need to include

```
import os  # isort:skip
import sys  # isort:skip
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # isort:skip
```

in preambule in order to be able to use our library as an external client (relative imports are not allowed in not top level scripts). Following [Python Guide](https://docs.python-guide.org/writing/structure/), section `Test Suite`.
