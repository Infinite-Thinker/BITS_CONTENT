import zipfile
import xml.etree.ElementTree as ET

def get_docx_text(path):
    try:
        doc = zipfile.ZipFile(path)
        xml_content = doc.read('word/document.xml')
        root = ET.fromstring(xml_content)
        
        # Word document namespaces
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        # We can traverse paragraph-by-paragraph or table-by-table. Let's do a general traversal
        # to ensure we capture tables, paragraphs, and other text elements in document order.
        # Actually, let's find all text elements or paragraphs and tables.
        # A simple paragraph-based search captures paragraphs:
        for paragraph in root.iter('{' + namespaces['w'] + '}p'):
            p_text = []
            for run in paragraph.iter('{' + namespaces['w'] + '}r'):
                for t in run.iter('{' + namespaces['w'] + '}t'):
                    if t.text:
                        p_text.append(t.text)
            text.append("".join(p_text))
        return "\n".join(text)
    except Exception as e:
        return f"Error: {e}"

text = get_docx_text("C:\\BITS_WILP\\ML_Assignment_1\\ML_Assignment1_AIML_2026.docx")
with open("ML_Assignment1_Requirements.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"Extracted {len(text)} characters to ML_Assignment1_Requirements.txt")
