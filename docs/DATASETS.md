# Data Sources

All data files live under `data/` (gitignored). Run `bash scripts/setup_data.sh` to download everything.

> **⚠️ Licensing disclaimer.** The license summaries below are best-effort and were
> compiled with AI assistance. They are **not legal advice**, may be incomplete or
> out of date, and have not been verified by a lawyer. Each source is governed solely
> by its own upstream terms — follow the links and confirm the current license yourself
> before relying on this data, **especially for commercial use**. You are responsible
> for verifying you have the rights you need.

## xd Clues Archive

- **Source**: [xd project](https://github.com/century-arcade/xd) by Saul Pwanson
- **URL**: https://xd.saul.pw/data/xd-clues.zip
- **Size**: 67MB zip → 242MB TSV
- **Contents**: ~7.5M crossword clue/answer pairs from historical puzzles
- **Path**: `data/sources/xd/clues.tsv`
- **License**: None stated; clue data extracted from published puzzles. See disclaimer.

## CrosswordQA

- **Source**: [albertxu/CrosswordQA](https://huggingface.co/datasets/albertxu/CrosswordQA) on HuggingFace
- **Size**: train.csv (246MB) + valid.csv (14MB)
- **Contents**: ~6.8M clue/answer pairs (academic dataset)
- **Path**: `data/sources/crosswordqa/train.csv`, `data/sources/crosswordqa/valid.csv`
- **License**: Unspecified ("unknown" on HuggingFace). See disclaimer.
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
- **License**: No formal license; author permits free/commercial use, asks you not to resell. See disclaimer.
- **Note**: Site has expired SSL cert; downloaded over HTTP

## SQLite Database (derived)

- **Path**: `data/clues.db`
- **Size**: ~1.4GB
- **Contents**: ~9-11M deduplicated clue/answer pairs from xd + CrosswordQA
- **Built by**: `make build-db` (or auto-built on first solver run)
- **Rebuild**: Delete `data/clues.db` and re-run `make build-db`
