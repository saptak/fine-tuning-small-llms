#!/bin/bash

# Create repository structure
mkdir -p part1-setup/{src,configs,scripts,docs}
mkdir -p part2-data-preparation/{src,configs,scripts,docs,examples}
mkdir -p part3-training/{src,configs,scripts,docs,notebooks}
mkdir -p part4-evaluation/{src,configs,scripts,docs,tests}
mkdir -p part5-deployment/{src,configs,scripts,docs,docker}
mkdir -p part6-production/{src,configs,scripts,docs,monitoring}

# Create shared directories
mkdir -p {data,models,logs,backups}
mkdir -p docker/{images,compose}
mkdir -p docs/{images,guides}

echo "Repository structure created successfully!"
