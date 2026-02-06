---
name: file-management
description: File management operations reference including reading, writing, organizing, and batch processing files.
---

# File Management Operations Reference

Comprehensive reference for file system operations and file management tasks.

## File Reading

### Bash
```bash
cat /path/file.txt                 # Full file
head -n 20 file.txt                # First N lines
tail -n 20 file.txt                # Last N lines
sed -n '10,20p' file.txt           # Line range
```

### Python
```python
with open('/path/file.txt', 'r') as f:
    content = f.read()             # Full file
    lines = f.readlines()          # As list
    for line in f: process(line)   # Line by line
```

## File Writing

### Bash
```bash
echo "content" > file.txt          # Overwrite
echo "content" >> file.txt         # Append
cat > file.txt << 'EOF'
  multiline
EOF
cp /source/file.txt /dest/         # Copy
mv /old.txt /new.txt               # Move/rename
```

### Python
```python
with open('/path/file.txt', 'w') as f:
    f.write('content\n')            # Overwrite
with open('/path/file.txt', 'a') as f:
    f.write('content\n')            # Append

import shutil
shutil.copy2('/source.txt', '/dest.txt')
```

## Directory Operations

### Bash
```bash
ls -la /path                       # List
find /path -type f                 # Files only
find /path -name "*.txt"            # By extension
mkdir -p /path/nested/dir          # Create
rm -rf /path/dir                   # Delete
cp -r /source /dest                # Copy
```

### Python
```python
import os
from pathlib import Path
import shutil

files = os.listdir('/path')        # List
files = list(Path('/path').glob('*.txt'))
for root, dirs, files in os.walk('/path'):
    for file in files: process(file)

os.makedirs('/path', exist_ok=True)
shutil.rmtree('/path/dir')
```

## File Information

### Bash
```bash
[ -f file.txt ] && echo "exists"   # Check existence
du -h file.txt                     # Size
stat file.txt                      # Full info
find /path -type f | wc -l         # Count files
```

### Python
```python
import os
from pathlib import Path

os.path.exists('/path')            # Existence
os.path.getsize('/path')           # Size
os.stat('/path')                   # Full info
Path('/path').is_file() / .is_dir()

# Check if file/directory
os.path.isfile('/path/to/file.txt')
os.path.isdir('/path/to/dir')

# Get file size
os.path.getsize('/path/to/file.txt')
Path('/path/to/file.txt').stat().st_size

# Get modification time
os.path.getmtime('/path/to/file.txt')

# Get file extension
ext = os.path.splitext('/path/to/file.txt')[1]  # '.txt'
ext = Path('/path/to/file.txt').suffix  # '.txt'

# Get filename without extension
name = os.path.splitext('file.txt')[0]  # 'file'
name = Path('file.txt').stem  # 'file'
```

## Batch File Operations

### Process Multiple Files (Bash)
```bash
# Process all .txt files
for file in /path/to/dir/*.txt; do
    echo "Processing $file"
    cat "$file"
done

# Find and process files
find /path/to/dir -name "*.txt" -type f | while read file; do
    echo "Processing $file"
    # Do something with $file
done

# Rename multiple files
for file in /path/to/dir/*.txt; do
    mv "$file" "${file%.txt}.backup.txt"
done

# Delete old files (older than 7 days)
find /path/to/dir -type f -mtime +7 -delete
```

### Process Multiple Files (Python)
```python
import os
from pathlib import Path

# Process all .txt files
directory = Path('/path/to/dir')
for file in directory.glob('*.txt'):
    print(f"Processing {file}")
    with open(file, 'r') as f:
        content = f.read()
        # Process content

# Process files recursively
for file in directory.rglob('*.txt'):
    print(f"Processing {file}")

# Rename multiple files
for file in directory.glob('*.txt'):
    new_name = file.with_suffix('.backup.txt')
    file.rename(new_name)

# Delete old files
import time
from datetime import datetime, timedelta

cutoff = time.time() - (7 * 24 * 60 * 60)  # 7 days ago
for file in directory.rglob('*'):
    if file.is_file() and file.stat().st_mtime < cutoff:
        file.unlink()
```

## File Format Operations

### CSV Files
```python
import csv

# Read CSV
with open('/path/to/file.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['column_name'])

# Write CSV
data = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25}
]
with open('/path/to/file.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'age'])
    writer.writeheader()
    writer.writerows(data)
```

### JSON Files
```python
import json

# Read JSON
with open('/path/to/file.json', 'r') as f:
    data = json.load(f)

# Write JSON
data = {'key': 'value', 'number': 123}
with open('/path/to/file.json', 'w') as f:
    json.dump(data, f, indent=2)

# Append to JSON array
with open('/path/to/file.json', 'r') as f:
    data = json.load(f)
data.append({'new': 'item'})
with open('/path/to/file.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### Text File Processing
```bash
# Count lines
wc -l /path/to/file.txt

# Count words
wc -w /path/to/file.txt

# Remove duplicates
sort /path/to/file.txt | uniq > output.txt

# Merge files
cat file1.txt file2.txt > merged.txt

# Split large file
split -l 1000 largefile.txt chunk_
```

## File Compression

### Bash
```bash
# Create tar.gz archive
tar -czf archive.tar.gz /path/to/dir

# Extract tar.gz
tar -xzf archive.tar.gz

# Create zip
zip -r archive.zip /path/to/dir

# Extract zip
unzip archive.zip

# Compress single file
gzip file.txt  # Creates file.txt.gz
gunzip file.txt.gz  # Extracts to file.txt
```

### Python
```python
import tarfile
import zipfile
import gzip

# Create tar.gz
with tarfile.open('archive.tar.gz', 'w:gz') as tar:
    tar.add('/path/to/dir', arcname='dir')

# Extract tar.gz
with tarfile.open('archive.tar.gz', 'r:gz') as tar:
    tar.extractall('/dest/path')

# Create zip
with zipfile.ZipFile('archive.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in Path('/path/to/dir').rglob('*'):
        zipf.write(file, file.relative_to('/path/to'))

# Extract zip
with zipfile.ZipFile('archive.zip', 'r') as zipf:
    zipf.extractall('/dest/path')
```

## Best Practices

- Always use absolute paths when possible
- Check if files exist before operations
- Use `with` statements in Python for automatic file closing
- Handle exceptions for file operations
- Be careful with recursive delete operations
- Use appropriate file permissions
- Consider file locking for concurrent access
- Test file operations on copies first
