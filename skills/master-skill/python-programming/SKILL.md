---
name: python-programming
description: Python programming reference with examples for common tasks, file operations, data processing, and more.
---

# Python Programming Reference

Comprehensive reference for Python programming tasks and operations.

## File Operations

```python
# Read
with open('/path/file.txt', 'r') as f:
    content = f.read()              # Full file
    lines = f.readlines()           # As list
    for line in f: process(line)    # Line by line

# Write
with open('/path/file.txt', 'w') as f:
    f.write('content\n')            # Overwrite
with open('/path/file.txt', 'a') as f:
    f.write('content\n')            # Append

# JSON
import json
with open('file.json', 'r') as f:
    data = json.load(f)
with open('file.json', 'w') as f:
    json.dump(data, f, indent=2)

# CSV
import csv
with open('file.csv', 'r') as f:
    for row in csv.DictReader(f): process(row)
```

## Directory Operations

```python
import os
from pathlib import Path
import shutil

# Check existence
os.path.exists('/path')
Path('/path').exists()

# Create
os.makedirs('/path/dir', exist_ok=True)
Path('/path/dir').mkdir(parents=True, exist_ok=True)

# List
os.listdir('/path')
list(Path('/path').iterdir())

# List recursively
for root, dirs, files in os.walk('/path'):
    for file in files:
        filepath = os.path.join(root, file)
list(Path('/path').rglob('*.txt'))

# Delete
os.remove('file.txt')                  # File
shutil.rmtree('/path/dir')             # Directory
```

## String Operations

```python
text = "hello world"
text.upper() / .lower() / .capitalize()
text.replace('o', 'X')

# Split & join
parts = text.split(' ')            # ['hello', 'world']
joined = ', '.join(parts)

# Strip
text.strip() / .lstrip() / .rstrip()

# Check
'hello' in text
text.startswith('he') / .endswith('ld')
text.find('world')                 # Index or -1
```

## List Operations

```python
items = [1, 2, 3, 4, 5]

# Modify
items.append(6)                    # Add to end
items.insert(0, 0)                 # Insert at index
items.extend([7, 8])               # Add multiple
items.remove(3)                    # Remove value
items.pop() / .pop(0)              # Remove by index

# Process
squares = [x**2 for x in range(10)]
evens = [x for x in items if x % 2 == 0]
sorted(items) / items.sort()
filtered = filter(lambda x: x > 5, items)
```

## Dictionary Operations

```python
data = {'name': 'John', 'age': 30}

# Access
data['name']                       # Raises KeyError if missing
data.get('name', 'default')

# Modify
data['city'] = 'NYC'
data.update({'age': 31, 'country': 'USA'})
del data['age']
data.pop('city', None)

# Iterate
for key, value in data.items(): pass
for key in data.keys(): pass
for value in data.values(): pass

# Comprehension
squared = {x: x**2 for x in range(5)}
```

## HTTP Requests

```python
import urllib.request
import urllib.parse
import json

# GET
response = urllib.request.urlopen('https://api.example.com')
data = json.loads(response.read())

# GET with params
params = urllib.parse.urlencode({'q': 'test'})
url = f'https://api.example.com?{params}'

# POST
data = json.dumps({'key': 'value'}).encode()
req = urllib.request.Request(url, data=data,
    headers={'Content-Type': 'application/json'})
response = urllib.request.urlopen(req)

# With auth
req.add_header('Authorization', 'Bearer TOKEN')

# Error handling
try:
    response = urllib.request.urlopen(url)
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}")
except urllib.error.URLError as e:
    print(f"URL Error: {e.reason}")
```
data = urllib.parse.urlencode({'key': 'value'}).encode()
req = urllib.request.Request('https://api.example.com/post', data=data)
response = urllib.request.urlopen(req)

# With headers
req = urllib.request.Request('https://api.example.com/data')
req.add_header('Authorization', 'Bearer token')
response = urllib.request.urlopen(req)
```

## Error Handling

```python
# Try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Always executed")

# Multiple exceptions
try:
    # code
    pass
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
```

## Regular Expressions

```python
import re

# Search
match = re.search(r'pattern', text)
if match:
    print(match.group())

# Find all
matches = re.findall(r'\d+', text)  # Find all numbers

# Replace
new_text = re.sub(r'old', 'new', text)

# Split
parts = re.split(r'[,;]', text)

# Compile for reuse
pattern = re.compile(r'\d+')
matches = pattern.findall(text)
```

## Date and Time

```python
from datetime import datetime, timedelta

# Current date/time
now = datetime.now()
today = datetime.today()

# Format
now.strftime('%Y-%m-%d %H:%M:%S')

# Parse
dt = datetime.strptime('2024-01-01', '%Y-%m-%d')

# Arithmetic
tomorrow = now + timedelta(days=1)
last_week = now - timedelta(weeks=1)
```

## Common Patterns

```python
# Read file, process, write result
with open('input.txt', 'r') as f_in:
    lines = f_in.readlines()

processed = [line.strip().upper() for line in lines]

with open('output.txt', 'w') as f_out:
    f_out.write('\n'.join(processed))

# Count occurrences
from collections import Counter
words = ['apple', 'banana', 'apple', 'cherry']
counts = Counter(words)
# Counter({'apple': 2, 'banana': 1, 'cherry': 1})

# Group items
from itertools import groupby
data = [1, 1, 2, 2, 2, 3, 1, 1]
for key, group in groupby(data):
    print(f"{key}: {list(group)}")
```

## Best Practices

- Use `with` statements for file operations
- Use list/dict comprehensions for concise code
- Handle exceptions appropriately
- Use pathlib for path operations
- Use f-strings for string formatting: `f"Value: {var}"`
