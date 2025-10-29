# data/augmented

Outputs from the augmentation stage, organised by reduction method and augmentation strategy.

Structure:

```
data/augmented/<reduction_method>/<augmentation_method>/<writer>/<dataset>/<ratio>/
```

- Each augmentation uses the corresponding reduced artefacts as input and then writes its own copies.
- Writer behaviour matches the reduced stage (HybEA vs RDF formats).
- Remove directories or rerun with `--overwrite-existing` to regenerate a particular combination.
