# Examples

This directory contains example scripts demonstrating how to use DAKGEA components programmatically.

## Available Examples

### test_set_knowledge_graph.py

Demonstrates how to work with knowledge graphs using the SetKnowledgeGraph class.

**Usage:**
```bash
python examples/test_set_knowledge_graph.py
```

**What it does:**
- Reads a dataset using the BERT-INT reader
- Converts it to a SetKnowledgeGraph representation
- Iterates through all triples in the knowledge graph

**Code overview:**
```python
from augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.core import DatasetReaderFactory

# Read dataset
reader = DatasetReaderFactory.create_reader("bert_int")
dataset = reader.read("path/to/dataset")

# Create knowledge graph
skg = SetKnowledgeGraph()
skg = skg.from_dataset(dataset)

# Access triples
for triple in skg.triples((None, None, None)):
    print(triple)
```

---

## Creating Your Own Examples

When creating new examples:

1. **Keep them simple**: Focus on demonstrating one concept at a time
2. **Add documentation**: Include comments explaining what the code does
3. **Make them runnable**: Ensure examples work out of the box
4. **Update this README**: Add your example to the list above

Example template:
```python
#!/usr/bin/env python3
"""Brief description of what this example demonstrates."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import ...

def main():
    """Main example logic."""
    # Your example code here
    pass

if __name__ == "__main__":
    main()
```

---

## See Also

- [Main Documentation](../README.md)
- [Scripts Directory](../scripts/) - For production tools
- [Experiments Documentation](../experiments/) - For running full experiments
- [Source Code](../src/) - For implementation details
