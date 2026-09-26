import re
import os

def extract_questions_from_file(file_path):
    print(f"\n==========================================")
    print(f"FILE: {file_path}")
    print(f"==========================================")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # We want to print sections of interest, or we can just print the whole content if it's small,
    # or write a script that searches for specific question markers like "Q1", "Q2", "(1)", "(2)", etc.
    # Actually, let's look at the length. If we read the first 300 lines of some of these, we'll get most of it.
    # Let's print out the text of each file, but filtered to show questions and answers.
    # Let's write a smart parser that looks for lines starting with "Q", "(1)", "(2)", "Solution" or similar.
    lines = content.split("\n")
    for i, line in enumerate(lines[:350]):  # print up to 350 lines
        # highlight questions or print them
        print(line)

def main():
    # Let's inspect the 2025-2026 SEM 2 exam in full first
    extract_questions_from_file("extracted_txt/PQP/2025-2026_SEM_2_AIMLZC416_MFML_EC2.txt")

if __name__ == "__main__":
    main()
