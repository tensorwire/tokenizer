# CLAUDE.md — Tokenizer

## What This Is

BPE tokenizer in pure Go. Loads HuggingFace tokenizer.json and SentencePiece tokenizer.model formats. Zero external dependencies — stdlib only.

## Build

```bash
go build ./...
go test -v ./...     # requires testdata/gpt2/ (included)
```

## Architecture

- `tokenizer.go` — BPE and greedy encoding, GPT-2 pre-tokenization, byte-level BPE support

## Key Functions

```go
tok, _ := tokenizer.LoadTokenizer("path/to/model")
ids := tok.Encode("hello world")        // [31373 995]
text := tok.Decode(ids)                  // "hello world"
chunks := tokenizer.GPTPreTokenize(text) // GPT-2 word splitting
size := tok.VocabSize()                  // 50257 for GPT-2
```

## Test Data

`testdata/gpt2/` contains the GPT-2 tokenizer files (merges.txt, vocab.json, tokenizer.json) for testing.

## Related Packages

- `github.com/open-ai-org/mongoose` — GPU compute engine
- `github.com/open-ai-org/gguf` — Model serialization
