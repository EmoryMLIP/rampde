# RAMPDE Publication Readiness - Product Requirements Document

## Project Overview
**Goal**: Complete the publication preparation of RAMPDE (formerly rampde), a mixed-precision Neural ODE solver library for PyTorch, alongside academic paper submission.

**Current Status**: Repository has been cleaned up and cloned to new private GitHub repository. Ready for systematic package renaming and publication preparation.

**Repository**: https://github.com/EmoryMLIP/rampde (private)
**Working Directory**: `/local/scratch/lruthot/code/rampde/`

---

## Completed Tasks ✅

### Phase 1: Repository Cleanup
- [x] Removed 36+ debug/temporary files
- [x] Committed essential files in 5 logical commits
- [x] Updated .gitignore for better experiment handling
- [x] Clean git working tree achieved

### Phase 2: Repository Migration
- [x] Created new private GitHub repository: `EmoryMLIP/rampde`
- [x] Cloned cleaned repository to `/local/scratch/lruthot/code/rampde/`
- [x] Configured git remote to new repository

---

## Remaining Tasks 🚧

### Phase 3: Package Renaming Operation
**Priority**: HIGH | **Estimated Effort**: 4-6 hours

#### 3.1 Core Package Rename
- [ ] Rename directory: `rampde/` → `rampde/`
- [ ] Update `__init__.py` imports and package references
- [ ] Update internal cross-references between modules

#### 3.2 Configuration Files
- [ ] **pyproject.toml**: Update package name, URLs, descriptions
- [ ] **setup.py**: Update package references
- [ ] **_version.py**: Verify version handling still works

#### 3.3 Import Statement Updates (~2000+ files)
**Files requiring systematic updates:**
- All Python files in `examples/`, `tests/`, `demos/`, `experiments/`
- Change `from rampde import` → `from rampde import`
- Change `import rampde` → `import rampde`
- Update internal package references

#### 3.4 Documentation Updates
- [ ] **README.md**: All package name references and URLs
- [ ] **CLAUDE.md**: Project documentation
- [ ] Docstrings in all Python modules
- [ ] Code comments mentioning rampde

#### 3.5 Test and File Renaming
- [ ] `tests/core/test_rampde.py` → `tests/core/test_rampde.py`
- [ ] `tests/core/test_rampde_tuple.py` → `tests/core/test_rampde_tuple.py`
- [ ] `experiments/roundoff_experiment/plot_rampde_dynamic.py` → `plot_rampde_dynamic.py`
- [ ] Update all references to renamed files

#### 3.6 Metadata and URLs
- [ ] GitHub repository URLs in all configuration files
- [ ] Package metadata classifiers and keywords
- [ ] License references if any

---

### Phase 4: Testing and Validation
**Priority**: HIGH | **Estimated Effort**: 3-4 hours

#### 4.1 Package Installation Testing
- [ ] Test `pip install -e .` works in new environment
- [ ] Test `pip install -e ".[dev]"` installs all dependencies
- [ ] Verify optional dependencies work correctly

#### 4.2 Core Functionality Testing
- [ ] Run full test suite: `python tests/run_all_tests.py`
- [ ] Test core demos: `python demos/demo_speedup.py`
- [ ] Test example scripts in `examples/`
- [ ] Verify import statements work: `python -c "import rampde; print(rampde.__version__)"`

#### 4.3 Experiment Validation
- [ ] Test experiment runners work: `experiments/run_all_experiments.sh`
- [ ] Verify SLURM scripts reference correct package
- [ ] Test figure generation: `python demos/generate_speedup_figure.py`

---

---

## Technical Implementation Notes

### Environment Setup
- **Conda Environment**: `torch26` (pre-configured)
- **Python Version**: >=3.8
- **Key Dependencies**: torch>=2.0, numpy, torchdiffeq (optional)

### File Search Strategy
The renaming operation affects ~2000+ files. Use systematic approach:
```bash
# Find all files with rampde references
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.toml" -o -name "*.txt" -o -name "*.sh" \) -exec grep -l "rampde" {} \;

# Count total references
find . -type f -name "*.py" -exec grep -c "rampde" {} + | awk -F: '{sum += $2} END {print sum}'
```

### Testing Priorities
1. **Core Package Import**: `import rampde` must work
2. **API Compatibility**: All public APIs unchanged
3. **Numerical Accuracy**: Results match rampde exactly
4. **Performance**: No regression in benchmark performance
5. **Dependencies**: Optional dependencies handled gracefully

---

## Risk Mitigation

### High Risk Areas
- **Import Statements**: 2000+ files need systematic updates
- **Circular Dependencies**: Internal package references must be updated correctly
- **Test File Names**: Renamed test files must be discoverable by test runner
- **SLURM Scripts**: Experiment scripts must reference correct package

### Rollback Strategy
- Original repository preserved at `/local/scratch/lruthot/code/rampde/`
- Git history maintained until final publication
- Can revert to any commit during development

### Validation Checkpoints
- After each phase, verify: `python -c "import rampde; rampde.odeint"`
- Run core test: `python tests/core/test_rampde.py`
- Check package metadata: `pip show rampde`

---

## Success Criteria

### Technical Milestones
- [ ] Package installs and imports successfully as `rampde`
- [ ] All tests pass with new package name
- [ ] Numerical results identical to original rampde
- [ ] Performance benchmarks show no regression
- [ ] All experiment scripts execute without errors

### Publication Readiness
- [ ] Repository suitable for public release
- [ ] Paper figures reproducible from repository
- [ ] Clear installation and usage documentation
- [ ] Professional commit history and release tags
- [ ] Reserved PyPI package name

---

## Contact and Handoff Information
- **Original Package**: rampde (Mixed precision Neural ODE solvers)
- **Target Package**: rampde (R-A-M-P-D-E)
- **Paper Context**: Academic publication requiring reproducible code
- **Timeline**: End of week for public release

This PRD provides complete specifications for converting rampde to rampde while maintaining all functionality and preparing for academic publication.