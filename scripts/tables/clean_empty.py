import os

BASE_DIR = os.getenv("BASE_WCD")
EX1_DATA = os.path.join(BASE_DIR, "data/exp1")


def remove_empty_dirs(root: str):
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            print(f"Removed empty dir: {dirpath}")


def main():
    remove_empty_dirs(EX1_DATA)


if __name__ == "__main__":
    main()