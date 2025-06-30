def extract_text_from_pdf(file):
    try:
        # First try with PyPDF2 (fast)
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        if text.strip():
            return text
        else:
            raise Exception("Empty text from PyPDF2")
    except:
        # Fallback to pdfplumber (better with layouts)
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text
