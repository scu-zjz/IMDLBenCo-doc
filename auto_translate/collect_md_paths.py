# 第一步：收集Markdown路径 collect_md_paths.py
import os
import json
import argparse

def find_md_files(root_dir):
    md_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.md'):
                full_path = os.path.join(dirpath, filename)
                md_paths.append(os.path.relpath(full_path, root_dir))
    return md_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Source directory path')
    parser.add_argument('--output', default='markdown_paths.json', help='Output JSON path')
    args = parser.parse_args()

    md_files = find_md_files(args.source)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "source_dir": args.source,
            "files": md_files
        }, f, indent=2)

    print(f"Found {len(md_files)} Markdown files. Output saved to {args.output}")

if __name__ == '__main__':
    main()