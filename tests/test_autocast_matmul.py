import torch
import unittest

class TestAutocastMatmul(unittest.TestCase):
    def _test_matmul(self, device):
        # Set manual seed for reproducibility
        torch.manual_seed(42)
        
        # Create random matrices A and B on the specified device
        A = torch.randn(10, 10, device=device)
        B = torch.randn(10, 10, device=device)
        
        # Use autocast with float16
        with torch.autocast(device_type=device, dtype=torch.float16):
            C_autocast = A @ B
        
        # Manually cast A and B to float16 and compute the product
        C_manual = A.to(torch.float16) @ B.to(torch.float16)
        
        # Convert both results to float32 before comparing
        C_autocast_32 = C_autocast.float()
        C_manual_32 = C_manual.float()
        
        # print error
        print(C_autocast_32- C_manual_32)
        
        # Check that the two results are close within a tolerance
        self.assertTrue(
            torch.allclose(C_autocast_32, C_manual_32, atol=1e-3),
            f"Results differ on device {device}"
        )

    def test_cpu(self):
        self._test_matmul("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        self._test_matmul("cuda")

if __name__ == '__main__':
    unittest.main()