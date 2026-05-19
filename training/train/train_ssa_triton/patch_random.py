import re

with open("run_benchmarks_random.py", "r") as f:
    content = f.read()

# Replace BabyLucioleLM with RandomBaselineLM wrapper around DummyLM
pattern = r"class BabyLucioleLM\(LMBase\):.*?def run_evaluation\("

replacement = """from lm_eval.models.dummy import DummyLM

class RandomBaselineLM(DummyLM):
    \"\"\"Random prediction baseline using lm_eval DummyLM.\"\"\"

    def __init__(self, tokenizer_name: str, max_length: int = 2048):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self._max_length = max_length
        self._batch_size = 1

    @property
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            self.tokenizer_name, trust_remote_code=True, use_fast=False
        )

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cpu"

    def get_performance_summary(self) -> dict:
        return {
            "calls": 0, "input_tokens": 0, "prep_seconds": 0.0,
            "forward_seconds": 0.0, "output_seconds": 0.0, "total_seconds": 0.0,
            "avg_call_seconds": 0.0, "forward_tokens_per_second": 0.0,
        }

def run_evaluation("""

new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Fix model initialization in run_evaluation
init_pattern = r"model = BabyLucioleLM\([^\)]+\)"
init_replacement = """model = RandomBaselineLM(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )"""
new_content = re.sub(init_pattern, init_replacement, new_content, flags=re.DOTALL)

with open("run_benchmarks_random.py", "w") as f:
    f.write(new_content)

