import os
import zipfile

ZIP_NAME = "2026AJ05253_DNN_Assignment.zip"

FILES_TO_INCLUDE = [
    "README.md",
    "requirements.txt",
    "report.pdf",
    "results.csv",
    "problem1/problem1.ipynb",
    "problem2/problem2.ipynb",
    "problem2/generated_samples.txt",
]

FIGURE_DIRS = [
    "problem1/figures",
    "problem2/figures"
]

def make_zip():
    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    total_files = 0
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for fpath in FILES_TO_INCLUDE:
            if os.path.exists(fpath):
                z.write(fpath, arcname=fpath)
                print(f"Added: {fpath}")
                total_files += 1
            else:
                print(f"WARNING: File not found: {fpath}")

        for fdir in FIGURE_DIRS:
            if os.path.exists(fdir):
                for fname in sorted(os.listdir(fdir)):
                    if fname.endswith(".png"):
                        full_p = os.path.join(fdir, fname).replace("\\", "/")
                        z.write(full_p, arcname=full_p)
                        print(f"Added figure: {full_p}")
                        total_files += 1

    zip_size_bytes = os.path.getsize(ZIP_NAME)
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    print(f"\nCreated {ZIP_NAME}")
    print(f"Total files in archive: {total_files}")
    print(f"Archive size: {zip_size_mb:.2f} MB ({zip_size_bytes} bytes)")
    assert zip_size_mb <= 25.0, f"ZIP file exceeds 25 MB limit: {zip_size_mb:.2f} MB"
    print("ZIP verification PASSED: Size is well within the 25 MB limit!")

if __name__ == "__main__":
    make_zip()
