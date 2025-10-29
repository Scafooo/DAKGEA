# data/raw

This directory holds the original datasets exactly as downloaded. The reduction and augmentation stages always read from here and never modify its contents.

- Each dataset sits under a reader-specific subdirectory (e.g., `hybea/<dataset>/`).
- For HybEA data the expected layout is `attribute_data/` and `knowformer_data/` with the TSV/ILL files shipped upstream.
- Avoid editing files in place; instead refresh the folder when new raw dumps become available.
