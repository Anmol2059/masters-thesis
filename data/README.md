# Data

Place evaluation datasets here. Structure expected by benchmarks:

```
data/
└── fisher_callhome_es/
    ├── audio_001.wav
    ├── audio_001.txt   ← reference transcript
    └── ...
```

Fisher/Callhome Spanish corpus requires LDC license:  
https://catalog.ldc.upenn.edu/LDC96S35

After downloading, run:
```bash
python scripts/prepare_fisher.py --data-dir /path/to/ldc/corpus
```
