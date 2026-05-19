import re

with open("run_benchmarks_random.py", "r") as f:
    text = f.read()

pattern = """def get_task_list(task_string: str) -> list[str]:
    if task_string.lower() in BENCHMARK_GROUPS:
        return BENCHMARK_GROUPS[task_string.lower()]

    tasks = [t.strip().lower() for t in task_string.split(",") if t.strip()]
    invalid_tasks = [t for t in tasks if t not in AVAILABLE_BENCHMARKS]
    if invalid_tasks:
        logger.warning("Unknown tasks: %s", invalid_tasks)
        logger.info("Available tasks: %s", list(AVAILABLE_BENCHMARKS.keys()))
        logger.info("Available groups: %s", list(BENCHMARK_GROUPS.keys()))
        tasks = [t for t in tasks if t in AVAILABLE_BENCHMARKS]
    return tasks"""

replacement = """def get_task_list(task_string: str) -> list[str]:
    if task_string.lower() in BENCHMARK_GROUPS:
        return BENCHMARK_GROUPS[task_string.lower()]

    raw_items = [t.strip().lower() for t in task_string.split(",") if t.strip()]
    tasks = []
    invalid_tasks = []
    
    for item in raw_items:
        if item in BENCHMARK_GROUPS:
            tasks.extend(BENCHMARK_GROUPS[item])
        elif item in AVAILABLE_BENCHMARKS:
            tasks.append(item)
        else:
            invalid_tasks.append(item)
            
    if invalid_tasks:
        logger.warning("Unknown tasks or groups: %s", invalid_tasks)
        logger.info("Available tasks: %s", list(AVAILABLE_BENCHMARKS.keys()))
        logger.info("Available groups: %s", list(BENCHMARK_GROUPS.keys()))
        
    return list(dict.fromkeys(tasks)) # remove duplicates"""

# I need to handle either List[str] or list[str] just in case
text = re.sub(r'def get_task_list.*?return tasks', replacement.replace('list[str]', 'List[str]'), text, flags=re.DOTALL)

with open("run_benchmarks_random.py", "w") as f:
    f.write(text)
