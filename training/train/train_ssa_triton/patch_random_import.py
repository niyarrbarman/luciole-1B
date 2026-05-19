import re
with open("run_benchmarks_random.py", "r") as f:
    content = f.read()

pattern = """from lm_eval.models.dummy import DummyLM

class RandomBaselineLM(DummyLM):"""

replacement = """try:
    from lm_eval.models.dummy import DummyLM
except ImportError:
    DummyLM = object

class RandomBaselineLM(DummyLM):"""

content = content.replace(pattern, replacement)

with open("run_benchmarks_random.py", "w") as f:
    f.write(content)
