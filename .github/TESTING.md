# Testing Strategy for rampde

This document describes the GitHub Actions testing strategy for the rampde project.

## Current Workflows

### 1. CI - Fast Tests (`ci.yml`)

**Purpose**: Provide rapid feedback on every push and pull request

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**What it tests**:
- Core unit tests (CPU-based, fast execution)
  - `test_odeint.py` - Convergence order tests
  - `simple_gradient_test.py` - Basic gradient checks
  - `test_ode_gradients_simple.py` - Numerical gradient validation
- Package import validation

**Environment**:
- Python 3.11
- PyTorch 2.x (CPU version)
- Ubuntu latest

**Expected duration**: 2-3 minutes

**Why these tests**: These tests cover fundamental functionality without requiring GPU resources, providing quick feedback to catch obvious bugs immediately.

---

### 2. Code Quality (`quality.yml`)

**Purpose**: Ensure code packaging and structure integrity

**Triggers**:
- Pull requests to `main` or `develop` branches

**What it checks**:
- Package structure verification
- Import chain validation
- Version attribute accessibility
- Test discovery by pytest
- Common code quality issues (excessive print statements, TODO comments)

**Environment**:
- Python 3.11
- Minimal dependencies (torch CPU + package only)
- Ubuntu latest

**Expected duration**: 1-2 minutes

**Why this workflow**: Catches packaging issues, import problems, and basic code quality concerns before they reach the main branch.

---

### 3. Full Test Suite (`test-full.yml`)

**Purpose**: Comprehensive compatibility testing across Python and PyTorch versions

**Triggers**:
- Scheduled: Weekly on Mondays at 2 AM UTC
- Manual trigger via workflow_dispatch
- Push to `main` branch (optional)

**What it tests**:
- All CPU-compatible core unit tests
- Comparison tests with torchdiffeq
- Dtype preservation tests (CPU variants)
- Basic functionality integration test

**Matrix**:
- Python versions: 3.9, 3.11, 3.12
- PyTorch versions: 2.0.0, 2.4.0
- Excludes: Python 3.12 with PyTorch 2.0.0 (compatibility)

**Expected duration**: 15-20 minutes per matrix combination (6 combinations total)

**Why this workflow**: Ensures compatibility across supported versions without blocking daily development. Scheduled runs catch regressions without slowing down PR reviews.

---

## Test Categories and GPU Requirements

### Tests Run in Current Workflows (CPU-compatible)

✅ **Core Unit Tests** (no GPU required):
- `test_odeint.py` - Convergence and accuracy tests
- `simple_gradient_test.py` - Basic gradient functionality
- `test_ode_gradients_simple.py` - Numerical gradient validation

✅ **Comparison Tests** (no GPU required, needs torchdiffeq):
- `test_rampde.py` (CPU variants)
- `test_rampde_tuple.py` (CPU variants)

✅ **Dtype Tests** (partial - CPU variants only):
- `test_dtype_preservation.py` (test_float32_cpu, test_float64_cpu)

### Tests NOT Run Yet (GPU-dependent)

⏸️ **Gradient Quality Tests** (require GPU):
- `test_backward.py` - Taylor expansion gradient tests (all tests skip without CUDA)
- `test_adjoint_scaling.py` - Mixed precision scaling validation (skips without CUDA)

⏸️ **Performance Tests** (require GPU):
- `test_speed.py` - FP16 vs FP32 speedup benchmarks (skips without CUDA)
- `test_performance_regression.py` - Performance regression suite (uses cuda:0)
- `test_otflow_performance.py` - Complex ODE performance (uses cuda:0)

⏸️ **Dtype Tests** (GPU variants):
- `test_dtype_preservation.py` GPU tests (float16, bfloat16, etc.)
- `test_rampde.py` CUDA variants
- `test_rampde_tuple.py` CUDA variants

---

## Future: GPU Testing with Self-Hosted Runners

### Why GPU Testing Matters

GPU tests are essential for:
1. **Mixed precision validation**: float16/bfloat16 behavior differs significantly on GPU
2. **Dynamic scaler testing**: Loss scaling only matters for GPU training
3. **Gradient quality**: Autograd behavior can differ between CPU and GPU
4. **Performance regression**: Ensure optimizations don't break real-world GPU usage

### Options for GPU Testing

#### Option 1: GitHub-Hosted GPU Runners (Paid)
- **Pros**: Managed by GitHub, no infrastructure maintenance
- **Cons**: Expensive, limited availability
- **Cost**: ~$1.00-$3.00 per hour depending on GPU type
- **Setup**: Available for GitHub Team/Enterprise plans

#### Option 2: Self-Hosted Runners (Recommended)
- **Pros**: Use existing infrastructure, cost-effective for regular testing
- **Cons**: Requires setup and maintenance
- **Requirements**:
  - Machine with NVIDIA GPU
  - Linux (Ubuntu recommended)
  - Network access to GitHub
  - GitHub Actions runner agent installed

#### Option 3: SLURM Integration (Academic/HPC Context)

Your current setup uses SLURM for HPC job scheduling. While GitHub Actions doesn't natively support SLURM, you can bridge them:

**Approach A: Self-hosted runner on SLURM login node**
- Install GitHub Actions runner on a SLURM login/head node
- Runner submits jobs to SLURM queue via `sbatch`
- Workflow waits for job completion and retrieves results
- **Pros**: Leverages existing HPC infrastructure
- **Cons**: Need persistent runner process, potential firewall issues

**Approach B: Webhook-triggered SLURM jobs**
- Set up webhook listener on HPC system
- GitHub Actions trigger webhook on specific events
- SLURM jobs run and post results back to GitHub API
- **Pros**: No persistent runner needed
- **Cons**: More complex setup, requires network configuration

**Approach C: Periodic sync**
- Scheduled SLURM jobs pull latest code and run tests
- Results posted to GitHub via API or stored artifacts
- **Pros**: Simple, no inbound network requirements
- **Cons**: Not real-time, less integrated with PR workflow

### Recommended GPU Testing Strategy

When GPU runners become available:

1. **Create `ci-gpu.yml` workflow**:
   - Runs on self-hosted GPU runner
   - Triggers: Manual or scheduled (nightly)
   - Tests: All GPU-dependent tests
   - Duration: ~15-20 minutes

2. **Create `performance.yml` workflow**:
   - Runs on self-hosted GPU runner with consistent hardware
   - Triggers: Weekly or before releases
   - Tests: Performance regression suite
   - Stores baseline metrics for comparison
   - Duration: ~30-45 minutes

3. **Update matrix testing** (optional):
   - Add GPU variant to `test-full.yml` matrix
   - Test CUDA 11.x and 12.x compatibility
   - Test different GPU architectures if available

### Setting Up Self-Hosted Runner (Future Reference)

```bash
# On your GPU machine (Ubuntu example)
cd /opt/actions-runner

# Download and extract runner (check latest version)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure runner (requires GitHub repo token)
./config.sh --url https://github.com/EmoryMLIP/rampde --token YOUR_TOKEN

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start

# Add GPU labels for workflow targeting
# In GitHub: Settings > Actions > Runners > Edit labels
# Add labels: self-hosted, linux, gpu, cuda
```

### Example GPU Workflow (Future)

```yaml
name: GPU Tests

on:
  workflow_dispatch:
  schedule:
    - cron: '0 3 * * *'  # Nightly at 3 AM

jobs:
  gpu-tests:
    runs-on: [self-hosted, linux, gpu]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install dependencies
      run: |
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        pip install -e ".[testing]"

    - name: Verify GPU
      run: |
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

    - name: Run GPU tests
      env:
        RAMPDE_TEST_QUIET: 1
      run: |
        python -m pytest tests/core/test_backward.py -v
        python -m pytest tests/core/test_adjoint_scaling.py -v
        python -m pytest tests/core/test_speed.py -v
        python -m pytest tests/core/test_dtype_preservation.py -v -k "cuda"
        python -m pytest tests/core/test_rampde.py -v -k "cuda"
```

---

## Monitoring and Maintenance

### Workflow Health Checks

Monitor workflow success rates:
- **CI workflow**: Should pass >95% of the time
- **Quality checks**: Should pass 100% (failures indicate packaging issues)
- **Full test suite**: Check weekly results for regressions

### Updating Test Selection

As the codebase evolves:
1. Add new fast CPU tests to `ci.yml`
2. Add comprehensive tests to `test-full.yml`
3. Keep GPU tests separate until runners are available
4. Update this document when test categories change

### Dependency Updates

Update workflows when:
- Adding new Python version support (update matrix in `test-full.yml`)
- Updating PyTorch minimum version (update all workflows)
- Adding new required dependencies (update install steps)

---

## Summary

**Current Coverage** (CPU-based):
- ✅ Fast feedback on every PR (2-3 minutes)
- ✅ Code quality and packaging validation
- ✅ Weekly comprehensive compatibility testing
- ✅ Core functionality and comparison tests

**Future Expansion** (with GPU runners):
- ⏸️ Mixed precision gradient validation
- ⏸️ Performance regression testing
- ⏸️ Full dtype preservation testing
- ⏸️ CUDA-specific test coverage

**Status Badges**: Added to README.md for visibility

This strategy balances speed, cost, and coverage while keeping the door open for GPU testing when infrastructure becomes available.
