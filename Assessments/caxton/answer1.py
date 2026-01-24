import os

def files_folders(path: str) -> None:
    """
    Traverses the folder and prints the contents starting with the deepest level first.

    Args: path (str) - The path of the root folder to traverse
    Returns: None
    """
    if not os.path.exists(path):
        print(f"The path {path} does not exist. Please check path")
        return

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            print(f"(File) {os.path.join(root, name)}")
        for name in dirs:
            print(f"(Folder) {os.path.join(root, name)}")


if __name__ == "__main__":
    files_folders("sleeping_bets")