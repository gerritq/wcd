import emoji
import ray
import os

"""
- https://docs.ray.io/en/latest/ray-core/handling-dependencies.html
    - put files in the same dir
    - source code in the dir available to all workers
    - when running on a clsuter uv run --active main.py
    - need to make sure ray/python versions of the env and cluster are the same
    - use directory flag if u do not wanna use the current working dir
- https://www.anyscale.com/blog/uv-ray-pain-free-python-dependencies-in-clusters#end-to-end-example-for-using-uv
    - set a flag: export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook




"""
# BASE_DIR = os.getenv("BASE_WCD")
# ray.init(runtime_env={"working_dir": BASE_DIR, 
#                         "excludes": [".git", "__pycache__", "data", 
#                                      "outputs", "models" ,"trainer_output",
#                                      ],
#                         })


BASE_DIR = os.getenv("BASE_WCD")
ray.init(num_cpus=2,
         runtime_env={"working_dir": BASE_DIR,
                      "excludes": [".git", "__pycache__", "data", 
                                   "outputs", "models" ,"trainer_output",],
        })
@ray.remote
def f():
    return emoji.emojize('Python is :thumbs_up:')

# Execute 1000 copies of f across a cluster.
print(ray.get([f.remote() for _ in range(2)]))