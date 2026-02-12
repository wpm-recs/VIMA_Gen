from vima_bench import make

env = make(task_name="visual_manipulation")

obs = env.reset()
prompt, prompt_assets = env.prompt, env.prompt_assets