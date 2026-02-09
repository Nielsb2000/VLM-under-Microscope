---
name: bash-scripting
description: Bash and shell scripting reference with commands and examples for file operations, text processing, and system tasks.
---

# Bash/Shell Scripting Reference

Comprehensive reference for bash commands and shell scripting operations.

## File Operations

```bash
# Read files
cat /path/to/file.txt              # Full file
head -n 10 file.txt                # First N lines
tail -n 10 file.txt                # Last N lines
sed -n '10,20p' file.txt           # Line range

# Write files
echo "content" > file.txt          # Overwrite
echo "content" >> file.txt         # Append
cat > file.txt << 'EOF'
  multiline
EOF

# File info
[ -f file.txt ] && echo "exists"   # Check existence
wc -l file.txt                     # Line count
du -h file.txt                     # File size
```

## Directory Operations

```bash
mkdir -p /path/to/nested/dir       # Create (with parents)
ls -la /path/to/dir                # List contents
find /path -type f                 # Find files recursively
rm -rf /path/to/dir                # Delete
cp -r /source /dest                # Copy
mv /old /new                       # Move/rename
```

## Text Processing

```bash
# Search
grep "pattern" file.txt            # Find text
grep -n "pattern" file.txt         # With line numbers
grep -i "pattern" file.txt         # Case-insensitive

# Replace
sed -i 's/old/new/g' file.txt      # Replace all occurrences
sed -i 's|/old/path|/new/path|g' f # Use | for paths

# Extract & filter
awk '{print $1, $3}' file.txt      # Extract columns
awk '$3 > 100' file.txt            # Filter by condition
sed -n '/START/,/END/p' file.txt   # Extract range
sed '/pattern/d' file.txt          # Remove matching lines
```



## Control Flow

```bash
# Loops
for file in *.txt; do process "$file"; done
while IFS= read -r line; do echo "$line"; done < file.txt

# Conditionals
if [ -f file.txt ]; then echo "exists"; fi
[ -f file.txt ] && echo "exists" || echo "not found"
```

## Variables

```bash
VAR="value"                        # Set
echo "$VAR"                        # Use
RESULT="${VAR}_suffix"             # Concatenate
echo "${#VAR}"                     # Length
echo "${VAR:0:5}"                  # Substring
echo "${VAR/old/new}"              # Replace
```

## Utilities

```bash
sort file.txt | uniq               # Unique lines
sort -u file.txt                   # Sort + unique
sort file.txt | uniq -c            # Count occurrences

# Join files line by line
paste file1.txt file2.txt

# Split file into chunks
split -l 100 /path/to/file.txt chunk_
```

## Best Practices

- Always quote variables: `"$VAR"` not `$VAR`
- Use `[[ ]]` for conditions instead of `[ ]`
- Check command success: `command && echo "success" || echo "failed"`
- Use absolute paths when possible
- Test scripts with `bash -n script.sh` before running
