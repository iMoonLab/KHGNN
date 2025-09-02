# Configuration Guide

This document explains how to configure KHGNN for different environments and use cases.

## Working Directory Configuration

The configuration files use `wd` (working directory) to specify where results and cache files are stored. We provide several flexible options:

### Option 1: Automatic Current Directory (Default)

```yaml
wd: ${hydra:runtime.cwd}
```

This automatically uses the current working directory where you run the script. This is the default setting and works for most users.

### Option 2: Environment Variable

```yaml
wd: ${oc.env:KHGNN_HOME,${hydra:runtime.cwd}}
```

This checks for an environment variable `KHGNN_HOME` first, and falls back to the current directory if not set.

To use this:
```bash
# Set environment variable (optional)
export KHGNN_HOME=/path/to/your/khgnn/workspace

# Run training
uv run trans_train.py
```

### Option 3: Absolute Path

```yaml
wd: /absolute/path/to/your/workspace
```

For users who want to specify an exact path.

### Option 4: Relative Path

```yaml
wd: ./workspace
```

Uses a relative path from where you run the script.

## Directory Structure

The configuration creates the following directory structure:

```
${wd}/
├── cache/
│   ├── trans/          # Transductive learning results
│   │   └── ${task}/
│   │       └── YYYY-MM-DD_HH-MM-SS/
│   └── prod/           # Inductive learning results
│       └── ${task}/
│           └── YYYY-MM-DD_HH-MM-SS/
```

Where `${task}` is automatically generated as:
`${data.name}__${data.num_train}-${data.num_val}__noise-${data.ft_noise_level}__${model.name}-${model.p_min}-${model.p_max}`

## Configuration Parameters

### Data Configuration

```yaml
data:
  name: cora                    # Dataset name
  num_train: 20                 # Number of training samples per class
  num_val: 100                  # Number of validation samples
  ft_noise_level: 0.0          # Feature noise level
  self_loop: True              # Add self loops
  random_split: True           # Random data split
  test_ind_ratio: 0.2          # Test ratio (for inductive learning)
```

### Model Configuration

```yaml
model:
  name: kerhgnn                # Model type
  p_min: -0.5                  # Minimum kernel parameter
  p_max: 2.0                   # Maximum kernel parameter
  hid: 32                      # Hidden dimension
  num_layer: 2                 # Number of layers
  kernel_type: poly            # Kernel type: poly, apoly, mean
  mu: 1                        # Kernel scaling factor
  use_norm: True               # Use normalization
```

### Optimization Configuration

```yaml
optim:
  lr: 0.1                      # Learning rate for main parameters
  lr_p: 0.003                  # Learning rate for kernel parameters
```

## Supported Datasets

Available datasets (uncomment the one you want to use):

- `cora` - Cora citation network
- `pubmed` - PubMed citation network  
- `citeseer` - CiteSeer citation network
- `news20` - 20 Newsgroups
- `dblp4k-conf` - DBLP conference data
- `dblp4k-paper` - DBLP paper data
- `dblp4k-term` - DBLP term data
- `coauthorship_dblp` - DBLP coauthorship
- `imdb4k` - IMDB movie data
- `cooking200` - Recipe data

## Supported Models

Available models (uncomment the one you want to use):

- `kerhgnn` - Kernel Hypergraph Neural Network (our method)
- `hgnn` - Hypergraph Neural Network
- `hgnnp` - HGNN+
- `gcn` - Graph Convolutional Network
- `gat` - Graph Attention Network
- `unignn` - UniGNN
- `unigat` - UniGAT
- `hnhn` - HNHN

## Kernel Types

Available kernel types for KHGNN:

- `poly` - Polynomial kernel with learnable exponent
- `apoly` - Adaptive polynomial kernel
- `mean` - Mean aggregation kernel

## Usage Examples

### Basic Training

```bash
# Transductive learning with default config
uv run trans_train.py

# Inductive learning with default config
uv run prod_train.py
```

### Override Configuration

```bash
# Change dataset
uv run trans_train.py data.name=cora

# Change model parameters
uv run trans_train.py model.hid=64 model.num_layer=3

# Change kernel type
uv run trans_train.py model.kernel_type=poly model.p_min=-1.0 model.p_max=3.0

# Change learning rates
uv run trans_train.py optim.lr=0.01 optim.lr_p=0.001
```

### Multi-Experiment Runs

```bash
# Run multiple experiments with different parameters
uv run trans_multi_train.py

# Inductive multi-experiments
uv run prod_multi_exp.py
```

## Troubleshooting

### Common Issues

1. **Permission Error**: Make sure you have write permissions to the working directory
2. **Path Not Found**: Ensure the working directory exists or can be created
3. **Config Override**: Use Hydra's syntax for overriding: `key=value`

### Debug Mode

Add `--cfg job` to see the resolved configuration:

```bash
uv run trans_train.py --cfg job
```

### Logging

Hydra automatically creates logs in the output directory. Check the `.hydra/` folder in your run directory for detailed logs.

## Best Practices

1. **Use version control**: Keep track of configuration changes
2. **Document experiments**: Use meaningful experiment names
3. **Backup results**: The cache directory contains all experimental results
4. **Environment isolation**: Use virtual environments for reproducibility
