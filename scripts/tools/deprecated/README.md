# Deprecated Scripts

This directory contains scripts that are no longer actively maintained but kept for reference.

## Script Obsoleti

### Config Generation

#### `generate_rrea_configs.py`
- **Status**: Deprecated (Dec 2024)
- **Sostituito da**: `generate_massive_configs.py`
- **Motivo**: Il nuovo script supporta sia bert_int che rrea, eliminando duplicazione

### Config Migration (One-time scripts)

Questi script sono stati usati per migrare le configurazioni a nuovi formati. Sono stati eseguiti e non sono più necessari per operazioni normali.

#### `update_massive_configs.py`
- **Data esecuzione**: Nov 2024
- **Scopo**: Conversione flag `save` → `save_dataset` e `save_model`
- **Status**: Completato

#### `update_bert_int_configs_augmented_only_train.py`
- **Data esecuzione**: Dec 2024
- **Scopo**: Aggiornamento writer config per bert_int
- **Status**: Completato

#### `update_rrea_configs_augmented_only_train.py`
- **Data esecuzione**: Dec 2024
- **Scopo**: Aggiornamento writer config per rrea
- **Status**: Completato

## Note

Questi script sono mantenuti per:
1. Riferimento storico
2. Possibile riutilizzo della logica
3. Documentazione delle migrazioni effettuate

**Non eseguire questi script** a meno che non si sappia esattamente cosa fanno - potrebbero sovrascrivere configurazioni esistenti.
