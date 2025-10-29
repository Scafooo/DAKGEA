# data/reduced

Cache of datasets after the reduction stage. The experiment runner writes here according to the chosen writer formats.

Directory convention:

```
data/reduced/<reduction_method>/<writer>/<dataset>/<ratio>/
```

- `ratio` is the percentage tag (e.g. `10`, `25`, `100`).
- Hybea writer restores the original `attribute_data/` and `knowformer_data/` folders.
- RDF writer stores `graph_source.nt`, `graph_target.nt`, and `aligned_entities.tsv`.

You can safely delete subfolders to force regeneration on the next run (or use `--overwrite-existing`).
