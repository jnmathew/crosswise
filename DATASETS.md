# Data Sources

All data files live under `data/` (gitignored). Run `bash scripts/setup_data.sh` to download everything.

## xd Clues Archive

- **Source**: [xd project](https://github.com/century-arcade/xd) by Saul Pwanson
- **URL**: https://xd.saul.pw/data/xd-clues.zip
- **Size**: 67MB zip → 242MB TSV
- **Contents**: ~7.5M crossword clue/answer pairs from historical puzzles
- **Path**: `data/xd/clues.tsv`
- **License**: Public archive of published crossword data

## CrosswordQA

- **Source**: [albertxu/CrosswordQA](https://huggingface.co/datasets/albertxu/CrosswordQA) on HuggingFace
- **Size**: train.csv (246MB) + valid.csv (14MB)
- **Contents**: ~6.8M clue/answer pairs (academic dataset)
- **Path**: `data/crosswordqa/train.csv`, `data/crosswordqa/valid.csv`
- **License**: Academic research dataset
- **Note**: Deduplicated against xd on (answer, clue_normalized) pairs during DB build

## Crossword Nexus Collaborative Word List

- **Source**: [Crossword-Nexus/collaborative-word-list](https://github.com/Crossword-Nexus/collaborative-word-list)
- **URL**: https://raw.githubusercontent.com/Crossword-Nexus/collaborative-word-list/main/xwordlist.dict
- **Size**: 8MB
- **Contents**: Scored word list for crossword construction/solving
- **Path**: `data/wordlists/xwordlist.dict`
- **License**: MIT

## Peter Broda Wordlist

- **Source**: [peterbroda.me/crosswords/wordlist](https://peterbroda.me/crosswords/wordlist/)
- **URL**: http://peterbroda.me/crosswords/wordlist/lists/peter-broda-wordlist__gridtext__scored__july-25-2023.txt
- **Size**: 8MB
- **Contents**: Scored word list for crossword construction/solving
- **Path**: `data/wordlists/peter-broda-wordlist__gridtext__scored__july-25-2023.txt`
- **License**: Freely distributed
- **Note**: Site has expired SSL cert; downloaded over HTTP

## SQLite Database (derived)

- **Path**: `data/clues.db`
- **Size**: ~1.4GB
- **Contents**: ~9-11M deduplicated clue/answer pairs from xd + CrosswordQA
- **Built by**: `make build-db` (or auto-built on first solver run)
- **Rebuild**: Delete `data/clues.db` and re-run `make build-db`
