# tokenizer

> **This project has moved to [github.com/tensorwire/tokenizer](https://github.com/tensorwire/tokenizer).** Updates will be published under [github.com/tensorwire](https://github.com/tensorwire). Code remaining in open-ai-org is unmaintained.

BPE tokenizer in pure Go. Loads HuggingFace `tokenizer.json` and SentencePiece `tokenizer.model` formats. Zero dependencies.

## Install

```bash
go get github.com/open-ai-org/tokenizer
```

## Usage

```go
tok, _ := tokenizer.LoadTokenizer("/path/to/model")
ids := tok.Encode("The meaning of life is")
text := tok.Decode(ids)
vocab := tok.VocabSize()
```

## Supported Formats

- HuggingFace `tokenizer.json` (Qwen, Llama, Mistral, GPT-2, etc.)
- SentencePiece `tokenizer.model`
- GPT-2 `vocab.json` + `merges.txt`

## Features

- Byte-level BPE encoding
- GPT-2 pre-tokenization regex
- Special token handling (EOS, BOS, padding)
- Pure Go, no CGo, no Python

## License

MIT
