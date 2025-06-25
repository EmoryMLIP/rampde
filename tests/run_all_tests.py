import unittest
import sys
import os

# Discover and run all tests in this directory
if __name__ == "__main__":
    # Set environment variable to suppress printouts in tests
    os.environ["TORCHMPNODE_TEST_QUIET"] = "1"
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n==================== Test Summary ====================")
    print(f"Ran {result.testsRun} tests.")
    if not result.wasSuccessful():
        print(f"FAILED: {len(result.failures) + len(result.errors)} tests failed.")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}")
    else:
        print("All tests passed!")
    print("=====================================================")
