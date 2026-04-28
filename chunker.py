import os

def chunk_markdown_file(file_path, chunk_size=100000, overlap=1000):
    """
    Chunks a large markdown file into smaller pieces with a specified overlap 
    to prevent context loss between trading system logs and codebase snippets.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = []
    start = 0
    total_chars = len(content)

    while start < total_chars:
        # Find a safe breaking point (like a double newline) to avoid splitting code blocks
        end = start + chunk_size
        if end < total_chars:
            # Try to find a paragraph break to split cleanly
            split_point = content.rfind("\n\n", start, end)
            if split_point != -1 and split_point > start + (chunk_size // 2):
                end = split_point + 2
                
        chunks.append(content[start:end])
        start = end - overlap # Step back to create the overlap

    print(f"✅ Split {file_path} into {len(chunks)} chunks.")
    return chunks

# Usage example for your trading setup:
# chunks = chunk_markdown_file("/kaggle/working/progress.md", chunk_size=50000, overlap=2000)
# for i, chunk in enumerate(chunks):
#     print(f"Processing chunk {i+1}/{len(chunks)}...")
#     # Feed `chunk` to Llama 3.1 70B via your NVIDIA NIM endpoint here