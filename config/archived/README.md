# Archived Configuration Files

This directory contains old configuration files kept for reference.

## Files

### `global_BKP.yaml`
- **Date**: November 26, 2024
- **Description**: Backup of global config with experimental `auto_retry_until_improvement` feature
- **Status**: Feature removed from current codebase

#### Feature: Auto-retry Until Improvement

This backup contains configuration for an experimental feature that would automatically retry augmentation if results didn't improve over the baseline.

**Configuration:**
```yaml
auto_retry_until_improvement:
  enabled: false              # Enable auto-retry mechanism
  max_attempts: 5             # Maximum retry attempts
  metric: "hits@1"            # Metric to compare
  min_improvement: 0.01       # Minimum improvement threshold (1%)
  save_all_attempts: true     # Save results from all attempts
```

**Why removed:**
- Feature was experimental and not fully tested
- Added complexity to the runner
- Better to handle retries manually in scripts

**Code reference:**
The implementation was in `experiments/runner/runner.py` around line 993, but may have been removed or refactored.

## Note

Files in this directory are **NOT** actively used by the system. They are kept for:
1. Historical reference
2. Documentation of removed features
3. Possible future restoration if needed
