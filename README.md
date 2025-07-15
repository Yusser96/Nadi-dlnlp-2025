# Nadi-dlnlp-2025

- path = os.path.join(base_dir,"./nadi2025_datasets", d ) # "Yemen")
- dataset = load_from_disk(path, keep_in_memory=True)["validation"]

| Country | Base (Whisper-large-v3) | Single-dialect-ft |
|---------|--------------------------|-------------------|
| Alg     | 89.95%                   | 56.09%            |
| Egy     | 58.31%                   | 34.91%            |
| Jor     | 45.34%                   | 30.80%            |
| Mau     | 96.19%                   | 60.19%            |
| Mor     | 91.96%                   | 51.44%            |
| Pal     | 57.69%                   | 32.72%            |
| UAE     | 61.86%                   | 35.49%            |
| Yem     | 70.44%                   | 55.28%            |
|---------|--------------------------|-------------------|
| Average | 71.47%                   | 44.37%            |


- dataset = load_dataset("UBC-NLP/Casablanca", d, split="test")

| Country | Base (Whisper-large-v3) | Single-dialect-ft |
|---------|--------------------------|-------------------|
| Alg     | 90.60%                   | 55.39%            |
| Egy     | 56.63%                   | 34.82%            |
| Jor     | 49.05%                   | 31.60%            |
| Mau     | 99.66%                   | 61.41%            |
| Mor     | 93.82%                   | 50.00%            |
| Pal     | 57.68%                   | 32.67%            |
| UAE     | 60.51%                   | 36.04%            |
| Yem     | 67.03%                   | 54.59%            |
|---------|--------------------------|-------------------|
| Average | 71.12%                   | 44.32%            |
