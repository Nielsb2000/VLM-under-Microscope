---
name: web-network
description: Web and network operations reference including HTTP requests, API calls, web scraping, and URL processing.
---

# Web & Network Operations Reference

Comprehensive reference for web requests, API interactions, and network operations.

## HTTP Requests with curl

```bash
# Basic requests
curl https://api.example.com
curl "https://api.example.com?q=test&limit=10"
curl -X POST https://api.example.com -H "Content-Type: application/json" -d '{"key":"val"}'
curl -X PUT / DELETE url

# Headers & auth
curl -H "Authorization: Bearer TOKEN" -H "Accept: application/json" url
curl -u username:password url

# Response handling
curl -o output.json url                    # Save to file
curl -s url                                # Silent
curl -s -o /dev/null -w "%{http_code}" url  # Status code only
curl -L url                                # Follow redirects
curl -i url                                # Show headers
curl -v url                                # Verbose

# Files
curl -O url                                # Download
curl -o custom.txt url                     # Custom name
curl -X POST url -F "file=@/path/file.txt" # Upload
```

## wget

```bash
wget https://example.com/file.txt          # Download
wget -O custom.txt url                     # Custom name
wget -r -np url                            # Recursive
wget -b url                                # Background
wget -c url                                # Resume
```

## Python HTTP (urllib)

```python
import urllib.request
import urllib.parse
import json

# GET
response = urllib.request.urlopen('https://api.example.com')
data = json.loads(response.read().decode())

# GET with params
params = urllib.parse.urlencode({'q': 'test'})
url = f'https://api.example.com?{params}'

# POST
data = json.dumps({'key': 'val'}).encode()
req = urllib.request.Request(url, data=data,
    headers={'Content-Type': 'application/json'})
response = urllib.request.urlopen(req)

# Auth
req.add_header('Authorization', 'Bearer TOKEN')

# Error handling
try:
    response = urllib.request.urlopen(url)
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}")
except urllib.error.URLError as e:
    print(f"URL Error: {e.reason}")
```

## Web Scraping (Basic)

```python
import urllib.request
import re

response = urllib.request.urlopen('https://example.com')
html = response.read().decode()

# Extract with regex
links = re.findall(r'href="(https?://[^"]+)"', html)
titles = re.findall(r'<title>(.*?)</title>', html)
headings = re.findall(r'<h1>(.*?)</h1>', html, re.DOTALL)
```

### HTML Parsing with html.parser
```python
from html.parser import HTMLParser
import urllib.request

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href':
                    self.links.append(value)

# Usage
response = urllib.request.urlopen('https://example.com')
html = response.read().decode()

parser = LinkParser()
parser.feed(html)
print(parser.links)
```

## URL Processing

```python
from urllib.parse import urlparse, urljoin, parse_qs, urlencode

# Parse URL
url = 'https://example.com/path?key=value&foo=bar'
parsed = urlparse(url)
print(parsed.scheme)   # 'https'
print(parsed.netloc)   # 'example.com'
print(parsed.path)     # '/path'
print(parsed.query)    # 'key=value&foo=bar'

# Parse query parameters
params = parse_qs(parsed.query)
print(params)  # {'key': ['value'], 'foo': ['bar']}

# Build URL with parameters
params = {'search': 'test', 'page': 1}
query_string = urlencode(params)
url = f'https://api.example.com/search?{query_string}'

# Join URLs
base = 'https://example.com/path/'
relative = '../other/page.html'
absolute = urljoin(base, relative)
```

## Common API Response Formats

### JSON Processing
```python
import json

# Parse JSON response
json_string = '{"name": "test", "value": 123}'
data = json.loads(json_string)

# Access nested data
data = {
    "results": [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"}
    ]
}
for item in data["results"]:
    print(f"{item['id']}: {item['name']}")

# Pretty print JSON
print(json.dumps(data, indent=2))
```

### XML Processing
```python
import xml.etree.ElementTree as ET

# Parse XML string
xml_string = '<root><item>value</item></root>'
root = ET.fromstring(xml_string)

# Find elements
for item in root.findall('item'):
    print(item.text)

# Parse XML file
tree = ET.parse('data.xml')
root = tree.getroot()
```

## Best Practices

- Always handle network errors (timeouts, connection errors)
- Set appropriate timeouts for requests
- Respect rate limits and use delays between requests
- Add User-Agent header to avoid blocks
- Use HTTPS when possible
- Validate and sanitize URLs
- Handle different response encodings
- Cache responses when appropriate
