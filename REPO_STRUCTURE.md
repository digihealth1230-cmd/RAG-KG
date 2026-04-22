# EKG-RAG Anonymous Repository - File Structure

```
ekg_rag_anonymous_submission/
├── README.md                          # Main repository documentation
├── USAGE.md                           # Detailed usage and reproduction guide
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── demo.py                           # End-to-end demo script
│
├── configs/
│   └── default.yaml                  # Hyperparameters matching Table 1
│
├── data/
│   ├── __init__.py
│   ├── README.md                     # Dataset download instructions
│   ├── meqsum_loader.py              # MeQSum dataset loader
│   └── mqp_loader.py                 # Medical Question Pair loader
│
├── retrieval/
│   ├── __init__.py
│   ├── splade_retriever.py           # SPLADE-v2 sparse retrieval + MMR
│   └── dense_reranker.py             # SapBERT dense re-ranking
│
├── models/
│   ├── __init__.py
│   └── train_qlora.py                # QLoRA fine-tuning script
│
├── evaluation/
│   ├── __init__.py
│   └── run_eval.py                   # Full evaluation loop (Table 2)
│
├── scripts/
│   └── run_all.sh                    # SLURM batch script for HPC
│
├── sapbert_extractor.py              # Clinical concept extraction + UMLS normalization
├── kg_builder.py                     # UMLS knowledge graph construction
├── medfaith_f1.py                    # MedFaith-F1 metric implementation
└── ekg_rag.py                        # Main EKG-RAG pipeline


