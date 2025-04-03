import argparse
import os
import tarfile
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor


def extract_tar(tar_path, dest_dir):
    """
    Extracts a .tar file to the specified destination directory.
    """
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=dest_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract dataset.")
    parser.add_argument("--local_dir", type=str, default="/mnt/localssd/",
                        help="Local directory to save the dataset.")
    parser.add_argument("--repo_id", type=str,
                        default="Languagebind/Open-Sora-Plan-v1.1.0", help="Hugging Face repository ID.")
    parser.add_argument("--folder_name", type=str, default="all_mixkit",
                        help="Folder name of the huggingface repo.")

    args = parser.parse_args()

    allow_patterns = [f"{args.folder_name}/*.tar"]
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        resume_download=True,
        local_dir_use_symlinks=False,
        revision="main",          # or the branch/tag/commit you want
        allow_patterns=allow_patterns,
        repo_type="dataset"
    )

    # 4. Collect all .tar files recursively from the downloaded folder
    tar_files = []
    for root, dirs, files in os.walk(args.local_dir):
        for file in files:
            if file.endswith(".tar"):
                tar_files.append(os.path.join(root, file))

    # 5. Destination folder for extracted files
    output_dir = os.path.join(args.local_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)

    # 6. Extract each tar file in parallel
    with ThreadPoolExecutor() as executor:
        for tar_path in tar_files:
            executor.submit(extract_tar, tar_path, output_dir)

    print("All .tar files have been downloaded and extracted to:", output_dir)


if __name__ == "__main__":
    main()
