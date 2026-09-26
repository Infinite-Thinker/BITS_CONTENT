import os
import pypdf

def extract_text_from_pdfs():
    output_dir = "extracted_text"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files.")
    
    for pdf_file in pdf_files:
        txt_filename = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
        print(f"Extracting {pdf_file} -> {txt_filename}...")
        try:
            reader = pypdf.PdfReader(pdf_file)
            num_pages = len(reader.pages)
            text_content = []
            for i in range(num_pages):
                page = reader.pages[i]
                text = page.extract_text() or ""
                text_content.append(f"=== PAGE {i+1} ===\n{text}\n")
            
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(text_content))
            print(f"Success! {num_pages} pages extracted.")
        except Exception as e:
            print(f"Error extracting {pdf_file}: {e}")

if __name__ == "__main__":
    extract_text_from_pdfs()
