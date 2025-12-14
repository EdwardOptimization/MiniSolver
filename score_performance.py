#!/usr/bin/env python3
"""
Performance Scoring Script for MiniSolver
==========================================

This script performs the following steps:
1. Runs ./build.sh to ensure the project is compiled
2. Runs the memory test (build/test_memory) to ensure no malloc is detected
3. If successful, runs the benchmark suite (build/benchmark_suite)
4. Parses the Time(ms) column from benchmark output
5. Outputs JSON result: {'status': 'success', 'avg_time': X.XX}

If any step fails, prints 'FAIL' and exits.
"""

import subprocess
import sys
import json
import re


def run_command(cmd, description, shell=False):
    """
    Run a shell command and return success status and output.
    
    Args:
        cmd: Command to run (string or list)
        description: Description for logging
        shell: Whether to use shell=True
        
    Returns:
        tuple: (success: bool, output: str)
    """
    print(f"[INFO] {description}...", file=sys.stderr)
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"[ERROR] {description} failed!", file=sys.stderr)
            print(f"[ERROR] stdout: {result.stdout}", file=sys.stderr)
            print(f"[ERROR] stderr: {result.stderr}", file=sys.stderr)
            return False, result.stdout + result.stderr
        
        return True, result.stdout
    
    except subprocess.TimeoutExpired:
        print(f"[ERROR] {description} timed out!", file=sys.stderr)
        return False, ""
    except Exception as e:
        print(f"[ERROR] {description} exception: {e}", file=sys.stderr)
        return False, ""


def main():
    """Main scoring workflow."""
    
    # Step 1: Build the project
    success, output = run_command(
        ["./build.sh"],
        "Building project with ./build.sh",
        shell=False
    )
    
    if not success:
        print("FAIL")
        sys.exit(1)
    
    print("[INFO] Build successful!", file=sys.stderr)
    
    # Step 2: Run memory test
    success, output = run_command(
        ["./build/test_memory"],
        "Running memory test (build/test_memory)",
        shell=False
    )
    
    if not success:
        print("FAIL")
        sys.exit(1)
    
    # Check if malloc was detected (test should pass with no allocations)
    # If the test passed, it means no malloc was detected during solve
    if "PASSED" not in output:
        print("[ERROR] Memory test did not pass - malloc may have been detected!", file=sys.stderr)
        print("FAIL")
        sys.exit(1)
    
    print("[INFO] Memory test passed (no malloc detected)!", file=sys.stderr)
    
    # Step 3: Run benchmark suite
    success, output = run_command(
        ["./build/benchmark_suite"],
        "Running benchmark suite (build/benchmark_suite)",
        shell=False
    )
    
    if not success:
        print("FAIL")
        sys.exit(1)
    
    print("[INFO] Benchmark suite completed!", file=sys.stderr)
    
    # Step 4: Parse Time(ms) values from output
    # The output format is a table with columns including "Time(ms)"
    # We need to extract all time values and calculate the average
    
    time_values = []
    
    # Look for lines with numerical data (skip header and separator lines)
    lines = output.split('\n')
    for line in lines:
        # Match lines that contain time data
        # Format example: "TURBO_MPC         Euler + Adaptive + Loose Tol       0.364       0.047  ..."
        match = re.search(r'^\s*\S+\s+.+?\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+', line)
        if match:
            time_ms = float(match.group(1))
            time_values.append(time_ms)
            print(f"[DEBUG] Found time: {time_ms} ms", file=sys.stderr)
    
    if not time_values:
        print("[ERROR] Could not parse any Time(ms) values from benchmark output!", file=sys.stderr)
        print("[ERROR] Output was:", file=sys.stderr)
        print(output, file=sys.stderr)
        print("FAIL")
        sys.exit(1)
    
    # Calculate average time
    avg_time = sum(time_values) / len(time_values)
    
    print(f"[INFO] Parsed {len(time_values)} time values", file=sys.stderr)
    print(f"[INFO] Average time: {avg_time:.2f} ms", file=sys.stderr)
    
    # Step 5: Output JSON result
    result = {
        'status': 'success',
        'avg_time': round(avg_time, 2)
    }
    
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
