with open("run_benchmarks_random.py", "r") as f:
    text = f.read()
text = text.replace("@property\n    def tokenizer(self):", "@cached_property\n    def tokenizer(self):\n        from functools import cached_property")
with open("run_benchmarks_random.py", "w") as f:
    f.write(text)
