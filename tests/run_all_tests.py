import unittest
import sys
import os
import subprocess
import argparse
import numpy as np
import torch

def set_deterministic_mode(seed=42):
    """Set deterministic mode for reproducible test results.

    Args:
        seed (int): Random seed to use for all random number generators.
    """
    print(f"Setting deterministic mode with seed={seed}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use deterministic algorithms (with warning only to avoid failures)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # Fallback for older PyTorch versions
        pass

    print("Deterministic mode enabled for reproducible testing")

def run_unittest_suite(test_dir, pattern="test_*.py"):
    """Run unittest suite for a directory."""
    print(f"Running tests in {test_dir}...")
    
    # Set environment variable to suppress printouts in tests
    os.environ["RAMPDE_TEST_QUIET"] = "1"
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_performance_tests():
    """Run performance regression tests."""
    print("Running performance regression tests...")
    
    performance_dir = os.path.join(os.path.dirname(__file__), "performance")
    
    # Run main performance regression test
    test_files = [
        "test_performance_regression.py",
        "test_otflow_performance.py"
    ]
    
    results = []
    for test_file in test_files:
        test_path = os.path.join(performance_dir, test_file)
        if os.path.exists(test_path):
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run([sys.executable, test_path], 
                                      capture_output=True, text=True, 
                                      cwd=performance_dir)
                results.append((test_file, result.returncode == 0, result.stdout, result.stderr))
            except Exception as e:
                results.append((test_file, False, "", str(e)))
        else:
            print(f"Warning: {test_file} not found")
    
    return results

def print_test_summary(unittest_result, performance_results=None):
    """Print comprehensive test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    # Unittest results
    if unittest_result:
        print(f"Unit Tests: Ran {unittest_result.testsRun} tests")
        if unittest_result.wasSuccessful():
            print("  ✓ All unit tests passed!")
        else:
            print(f"  ✗ {len(unittest_result.failures) + len(unittest_result.errors)} tests failed")
            
            if unittest_result.failures:
                print("\n  Failures:")
                for test, traceback in unittest_result.failures:
                    print(f"    - {test}")
            
            if unittest_result.errors:
                print("\n  Errors:")
                for test, traceback in unittest_result.errors:
                    print(f"    - {test}")
    
    # Performance test results
    if performance_results:
        print(f"Performance Tests: Ran {len(performance_results)} test suites")
        all_performance_passed = True
        
        for test_file, success, stdout, stderr in performance_results:
            if success:
                print(f"  ✓ {test_file}: PASSED")
            else:
                print(f"  ✗ {test_file}: FAILED")
                all_performance_passed = False
                if stderr:
                    print(f"    Error: {stderr[:200]}...")
        
        if all_performance_passed:
            print("  All performance tests passed!")
    
    # Overall result
    unittest_success = unittest_result.wasSuccessful() if unittest_result else True
    performance_success = (not performance_results or 
                          all(success for _, success, _, _ in performance_results))
    
    overall_success = unittest_success and performance_success
    
    print(f"\nOverall Result: {'PASS' if overall_success else 'FAIL'}")
    print("=" * 60)
    
    return overall_success

def main():
    """Main test runner with command line options."""
    parser = argparse.ArgumentParser(description="Run rampde test suite")
    parser.add_argument("--include-performance", action="store_true",
                      help="Include performance regression tests")
    parser.add_argument("--performance-only", action="store_true",
                      help="Run only performance tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Verbose output")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for deterministic testing (default: 42)")
    
    args = parser.parse_args()

    # Set deterministic mode for reproducible testing
    set_deterministic_mode(seed=args.seed)

    # Determine what tests to run
    run_unit_tests = not args.performance_only
    run_perf_tests = args.include_performance or args.performance_only

    print("\nrampde Test Suite")
    print("=" * 60)
    
    unittest_result = None
    performance_results = None
    
    # Run unit tests
    if run_unit_tests:
        # Run core tests from the new core directory
        core_test_dir = os.path.join(os.path.dirname(__file__), "core")
        if os.path.exists(core_test_dir):
            unittest_result = run_unittest_suite(core_test_dir)
        else:
            # Fallback to old structure
            test_dir = os.path.dirname(__file__)
            unittest_result = run_unittest_suite(test_dir)
    
    # Run performance tests
    if run_perf_tests:
        if unittest_result:
            print("\n" + "=" * 60)
        performance_results = run_performance_tests()
        
        if args.verbose and performance_results:
            print("\nPerformance Test Details:")
            for test_file, success, stdout, stderr in performance_results:
                print(f"\n{test_file}:")
                if stdout:
                    print(stdout)
                if stderr:
                    print(f"Errors: {stderr}")
    
    # Print summary
    success = print_test_summary(unittest_result, performance_results)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
