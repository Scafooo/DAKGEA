"""Debug script to test why 'name' predicates are not matching."""

# Test del predicate matching per capire perché name non matcha con name

src_literals = {
    "name": ("http://example.org/name", ["debbi peterson"])
}

tgt_literals = {
    "name": ("http://example.org/name", ["debbi peterson"]),
    "dateofbirth": ("http://example.org/dateofbirth", ["1961"]),
    "comment": ("http://example.org/comment", ["to lead singles..."]),
    "surname": ("http://example.org/surname", ["peterson"]),
    "id": ("http://example.org/id", ["677106"]),
    "abstract": ("http://example.org/abstract", ["to lead singles..."]),
    "givenname": ("http://example.org/givenname", ["debbi"])
}

print("Source predicates:", list(src_literals.keys()))
print("Target predicates:", list(tgt_literals.keys()))
print()
print("Expected match: name ↔ name")
print()

# Il problema è che 'name' appare in entrambi ma viene considerato unmatched
# Questo significa che il matching non funziona

# Possibili cause:
# 1. Threshold troppo alto
# 2. Cache di allineamento non contiene il match
# 3. Bug nella logica di matching

print("Analizziamo i log per capire cosa sta succedendo...")
print("Se non ci sono 'Found X Matching Predicate Pairs', allora il problema è nel matching")
