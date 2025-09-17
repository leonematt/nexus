#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
from pathlib import Path
import time
import re
import sys
import importlib.util
from typing import List, Dict, Any
import os

def compile_triton_variants(kernel, variants, output_dir="./ptx_output", rename_kernels=True):
  """
  Simple function to compile Triton kernel variants and extract PTX.
  
  Args:
    kernel: @triton.jit decorated function
    variants: List of dicts with constants, e.g. [{'BLOCK_SIZE': 256}, {'BLOCK_SIZE': 512}]
    output_dir: Where to save PTX files
    rename_kernels: If True, rename kernel functions in PTX to include constants
  
  Returns:
    List of dicts with compilation info
  """

  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  cache_dir = Path.home() / ".triton" / "cache"

  kernel_name = kernel.fn.__name__
  results = []

  print(f"Compiling {len(variants)} variants of {kernel_name}")

  for i, constants in enumerate(variants):
    const_str = "_".join(f"{k}{v}" for k, v in sorted(constants.items()))
    variant_name = f"{kernel_name}_{const_str}"

    print(f"  [{i+1}/{len(variants)}] {variant_name}")

    old_dirs = set()
    if cache_dir.exists():
      old_dirs = {d.name for d in cache_dir.iterdir() if d.is_dir()}

    try:
      compiled = kernel[constants]
      time.sleep(0.3)  # Wait for cache

      ptx_content = None
      original_mangled_name = "unknown"

      if cache_dir.exists():
        new_dirs = {d.name for d in cache_dir.iterdir() if d.is_dir()} - old_dirs

        search_dirs = list(new_dirs) if new_dirs else []
        if not search_dirs:
          recent_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir()], 
                             key=lambda x: x.stat().st_mtime, reverse=True)
          search_dirs = [d.name for d in recent_dirs[:3]]

        for dir_name in search_dirs:
          cache_path = cache_dir / dir_name
          ptx_files = list(cache_path.glob("*.ptx"))

          for ptx_file in ptx_files:
            if kernel_name in ptx_file.name:
              try:
                with open(ptx_file, 'r') as f:
                  ptx_content = f.read()

                matches = re.findall(r'\.visible\s+\.entry\s+(\w+)', ptx_content)
                if matches:
                  original_mangled_name = matches[0]
                break
              except:
                continue
          if ptx_content:
            break

      final_mangled_name = original_mangled_name
      if ptx_content and rename_kernels:
        new_mangled_name = variant_name

        modified_ptx = re.sub(
          rf'(\.visible\s+\.entry\s+){re.escape(original_mangled_name)}',
          rf'\1{new_mangled_name}',
          ptx_content
        )

        modified_ptx = re.sub(
          rf'\b{re.escape(original_mangled_name)}\b',
          new_mangled_name,
          modified_ptx
        )

        ptx_content = modified_ptx
        final_mangled_name = new_mangled_name

        print(f"    Renamed: {original_mangled_name} -> {new_mangled_name}")

      ptx_file = output_path / f"{variant_name}.ptx"
      if ptx_content:
        header = f"""// Triton kernel variant: {variant_name}
// Original kernel: {kernel_name}
// Constants: {constants}
// Original mangled name: {original_mangled_name}
// Final mangled name: {final_mangled_name}

"""
        with open(ptx_file, 'w') as f:
          f.write(header + ptx_content)
        print(f"    Created {ptx_file.name} ({len(ptx_content)} bytes)")
      else:
        with open(ptx_file, 'w') as f:
          f.write(f"""// Triton kernel variant: {variant_name}
// Original kernel: {kernel_name}
// Constants: {constants}
// Status: Compiled successfully but PTX not accessible in this Triton version
// Note: Check ~/.triton/cache/ for actual PTX files

// Placeholder PTX - replace with actual content
.version 7.0
.target sm_80
.address_size 64

.visible .entry {variant_name}(
  .param .u64 .ptr .global .align 8 param_0
)
{{
  // Actual implementation would be here
  ret;
}}
""")
        print(f"    Created placeholder {ptx_file.name}")

      results.append({
        'variant': variant_name,
        'constants': constants,
        'original_mangled_name': original_mangled_name,
        'final_mangled_name': final_mangled_name,
        'ptx_file': str(ptx_file),
        'source_file': None  # Will be set by caller
      })

    except Exception as e:
      print(f"    Failed: {e}")

  print(f"Compiled {len(results)} variants")
  return results

def load_kernel_module(file_path):
  """Load a Python file and extract kernels and variants from it."""

  file_path = Path(file_path)
  if not file_path.exists():
    raise FileNotFoundError(f"Kernel file not found: {file_path}")

  # Create a unique module name to avoid conflicts
  module_name = f"kernel_module_{file_path.stem}_{hash(str(file_path))}"
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  module = importlib.util.module_from_spec(spec)

  module.torch = torch
  module.triton = triton
  module.tl = tl

  try:
    spec.loader.exec_module(module)
  except Exception as e:
    raise ImportError(f"Failed to load kernel module {file_path}: {e}")

  kernels = {}
  variants = None
  run_func = None

  for name in dir(module):
    obj = getattr(module, name)

    if hasattr(obj, '__call__') and hasattr(obj, 'fn'):
      kernels[name] = obj

    elif name == 'variants' and isinstance(obj, list):
      variants = obj

    elif name == 'run' and callable(obj):
      run_func = obj

  return kernels, variants, run_func

def compile_kernel_file(file_path, output_dir="./ptx_output"):
  """
  Load a kernel file and compile all its variants.

  The kernel file should contain:
  1. @triton.jit decorated functions
  2. A 'variants' list with parameter combinations
  3. A 'run' function to populate the cache

  Example:
    @triton.jit
    def add_kernel(...):
      pass

    variants = [
      {'BLOCK_SIZE': 256},
      {'BLOCK_SIZE': 512}
    ]

    def run():
      # Execute kernels to populate cache
      pass
  """

  print(f"\nProcessing kernel file: {file_path}")

  try:
    kernels, variants, run_func = load_kernel_module(file_path)
  except Exception as e:
    print(f"  Error loading file: {e}")
    return []

  print(f"  Found kernels: {list(kernels.keys())}")
  print(f"  Found variants: {variants is not None}")
  print(f"  Found run function: {run_func is not None}")

  if not kernels:
    print("  No Triton kernels found in file")
    return []

  if variants is None:
    print("  No 'variants' list found in file")
    return []

  if run_func:
    print("  Running kernels to populate cache...")
    try:
      run_func()
      print("  Cache populated successfully")
    except Exception as e:
      print(f"  Run function failed: {e}")
      print("  Proceeding with compilation anyway...")
  else:
    print("  No run function found - cache may not be populated")

  all_results = []

  for kernel_name, kernel_func in kernels.items():
    print(f"\n  {'='*40}")
    results = compile_triton_variants(kernel_func, variants, output_dir)
    
    # Add source file info to results
    for result in results:
      result['source_file'] = str(file_path)
    
    all_results.extend(results)

  return all_results

def find_triton_files(directory):
  """
  Recursively find all Python files that might contain Triton kernels.
  
  Args:
    directory: Directory to search recursively
    
  Returns:
    List of Path objects for Python files
  """
  directory = Path(directory)
  if not directory.exists():
    raise FileNotFoundError(f"Directory not found: {directory}")
  
  if not directory.is_dir():
    raise NotADirectoryError(f"Path is not a directory: {directory}")
  
  # Find all Python files recursively
  python_files = list(directory.rglob("*.py"))
  
  # Filter out common non-kernel files
  excluded_patterns = [
    "__pycache__",
    ".git",
    ".pytest_cache",
    "test_",
    "_test.py",
    "setup.py",
    "__init__.py"
  ]
  
  filtered_files = []
  for file_path in python_files:
    # Skip files in excluded directories or with excluded patterns
    skip = False
    for pattern in excluded_patterns:
      if pattern in str(file_path):
        skip = True
        break
    
    if not skip:
      filtered_files.append(file_path)
  
  return filtered_files

def compile_directory(directory, output_dir="./ptx_output"):
  """
  Recursively find and compile all Triton kernel files in a directory.
  
  Args:
    directory: Root directory to search
    output_dir: Where to save PTX files
    
  Returns:
    Dict with compilation summary
  """
  
  print(f"Searching for Triton kernel files in: {directory}")
  
  try:
    python_files = find_triton_files(directory)
  except Exception as e:
    print(f"Error searching directory: {e}")
    return {'files_processed': 0, 'files_with_kernels': 0, 'total_variants': 0, 'results': []}
  
  print(f"Found {len(python_files)} Python files to examine")
  
  if not python_files:
    print("No Python files found in directory")
    return {'files_processed': 0, 'files_with_kernels': 0, 'total_variants': 0, 'results': []}
  
  all_results = []
  files_with_kernels = 0
  
  for file_path in python_files:
    print(f"\n{'='*60}")
    results = compile_kernel_file(file_path, output_dir)
    
    if results:
      files_with_kernels += 1
      all_results.extend(results)
  
  summary = {
    'files_processed': len(python_files),
    'files_with_kernels': files_with_kernels,
    'total_variants': len(all_results),
    'results': all_results
  }
  
  return summary

def main():
  """Main entry point - compile all kernel files in specified directory."""

  if len(sys.argv) != 2:
    print("Usage: python triton_compiler.py <directory>")
    print("\nExample:")
    print("  python triton_compiler.py ./kernels/")
    print("  python triton_compiler.py /path/to/triton/kernels/")
    print("\nThe script will recursively search for Python files containing:")
    print("  - @triton.jit decorated functions")
    print("  - A 'variants' list with parameter combinations")
    print("  - A 'run' function to populate the cache")
    sys.exit(1)

  directory = sys.argv[1]
  
  # Create output directory based on input directory name
  input_dir_name = Path(directory).resolve().name
  output_dir = f"./ptx_{input_dir_name}"

  try:
    summary = compile_directory(directory, output_dir)

    print(f"\n{'='*80}")
    print(f"COMPILATION SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {summary['files_processed']}")
    print(f"Files with kernels: {summary['files_with_kernels']}")
    print(f"Total variants compiled: {summary['total_variants']}")
    print(f"Output directory: {output_dir}")
    
    if summary['results']:
      print(f"\nCompiled variants:")
      
      # Group results by source file for better organization
      by_file = {}
      for result in summary['results']:
        source_file = result['source_file']
        if source_file not in by_file:
          by_file[source_file] = []
        by_file[source_file].append(result)
      
      for source_file, results in by_file.items():
        print(f"\n  {Path(source_file).relative_to(Path(directory))}:")
        for result in results:
          print(f"    {result['variant']}")
          print(f"      Original: {result['original_mangled_name']}")
          print(f"      Renamed:  {result['final_mangled_name']}")
          print(f"      File:     {Path(result['ptx_file']).name}")

      print(f"\nPTX files saved to: {output_dir}/")
    else:
      print("\nNo variants were compiled")
      print("Make sure your Python files contain:")
      print("  - @triton.jit decorated functions")
      print("  - A 'variants' list")
      print("  - A 'run' function")

  except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

if __name__ == "__main__":
  main()