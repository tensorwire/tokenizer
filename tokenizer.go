package tokenizer

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Tokenizer handles BPE tokenization for LLM inference.
// Supports HuggingFace tokenizer.json and sentencepiece tokenizer.model.
type Tokenizer struct {
	Vocab            map[string]int
	Inverse          map[int]string
	Merges           [][2]string    // BPE merge rules in priority order
	mergePriority    map[string]int // "a b" → rank, built once at load
	AddedTokens      []string       // special token strings (sorted longest first for greedy match)
	BOS              int            // beginning of sequence token ID
	EOS              int            // end of sequence token ID
	byteLevelPretok  bool           // use GPT-style word splitting before BPE
}

// LoadTokenizer loads a tokenizer from a model directory.
// Tries tokenizer.json first (HuggingFace), then tokenizer.model (sentencepiece).
func LoadTokenizer(modelDir string) (*Tokenizer, error) {
	// Try HuggingFace tokenizer.json first
	jsonPath := filepath.Join(modelDir, "tokenizer.json")
	if _, err := os.Stat(jsonPath); err == nil {
		return loadTokenizerJSON(jsonPath)
	}

	// Try sentencepiece tokenizer.model
	modelPath := filepath.Join(modelDir, "tokenizer.model")
	if _, err := os.Stat(modelPath); err == nil {
		return loadSentencePiece(modelPath)
	}

	return nil, fmt.Errorf("no tokenizer.json or tokenizer.model found in %s", modelDir)
}

// loadTokenizerJSON reads HuggingFace tokenizer.json format.
func loadTokenizerJSON(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var tj struct {
		Model struct {
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"` // []string or [][]string
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		PreTokenizer *struct {
			Type           string `json:"type"`
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern *struct {
					Regex string `json:"Regex"`
				} `json:"pattern"`
			} `json:"pretokenizers"`
			Pattern *struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, err
	}

	t := &Tokenizer{
		Vocab:   tj.Model.Vocab,
		Inverse: make(map[int]string),
		BOS:     -1, // -1 = no BOS (Qwen, etc.)
		EOS:     2,
	}

	for k, v := range t.Vocab {
		t.Inverse[v] = k
	}

	// Added tokens (special tokens)
	for _, at := range tj.AddedTokens {
		t.Vocab[at.Content] = at.ID
		t.Inverse[at.ID] = at.Content
		t.AddedTokens = append(t.AddedTokens, at.Content)
		if at.Content == "<s>" || at.Content == "<bos>" {
			t.BOS = at.ID
		}
		if at.Content == "</s>" || at.Content == "<eos>" {
			t.EOS = at.ID
		}
	}
	sort.Slice(t.AddedTokens, func(i, j int) bool {
		return len(t.AddedTokens[i]) > len(t.AddedTokens[j])
	})

	// Detect if this is a byte-level BPE tokenizer with pretokenization.
	// Instead of parsing the regex (Go's regexp lacks lookaheads), we use
	// a simple GPT-style splitter: split on spaces keeping the space with
	// the following word, handle contractions and punctuation.
	hasByteLevelPretok := false
	if tj.PreTokenizer != nil {
		for _, pt := range tj.PreTokenizer.Pretokenizers {
			if pt.Type == "ByteLevel" {
				hasByteLevelPretok = true
				break
			}
		}
		if tj.PreTokenizer.Type == "ByteLevel" {
			hasByteLevelPretok = true
		}
	}
	t.byteLevelPretok = hasByteLevelPretok

	// Detect tokenizer style: sentencepiece (▁) vs byte-level BPE (Ġ)
	// If vocab contains ▁-prefixed tokens, use sentencepiece greedy encoder.
	// If vocab contains Ġ-prefixed tokens, use byte-level BPE encoder.
	isSentencePieceStyle := false
	for tok := range t.Vocab {
		if strings.HasPrefix(tok, "▁") {
			isSentencePieceStyle = true
			break
		}
	}

	if !isSentencePieceStyle && len(tj.Model.Merges) > 0 {
		// Parse merge rules — handles two HuggingFace formats:
		//   Old: ["Ġ t", "h e", ...]        (space-separated strings)
		//   New: [["Ġ", "t"], ["h", "e"]]   (arrays of 2 strings)
		var mergesOld []string
		var mergesNew [][2]string
		if err := json.Unmarshal(tj.Model.Merges, &mergesOld); err == nil {
			// Old format: split on space
			for _, m := range mergesOld {
				parts := strings.SplitN(m, " ", 2)
				if len(parts) == 2 {
					t.Merges = append(t.Merges, [2]string{parts[0], parts[1]})
				}
			}
		} else if err := json.Unmarshal(tj.Model.Merges, &mergesNew); err == nil {
			// New format: already paired
			t.Merges = mergesNew
		}
	}
	// If sentencepiece style, Merges stays empty → Encode uses encodeGreedy

	if len(t.Merges) > 0 {
		t.mergePriority = make(map[string]int, len(t.Merges))
		for i, m := range t.Merges {
			t.mergePriority[m[0]+" "+m[1]] = i
		}
	}

	return t, nil
}

// loadSentencePiece reads a sentencepiece .model file.
// The file is a protobuf (ModelProto) but we only need the vocab.
// We parse just enough of the protobuf wire format to extract piece/score pairs.
func loadSentencePiece(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	t := &Tokenizer{
		Vocab:   make(map[string]int),
		Inverse: make(map[int]string),
		BOS:     1,
		EOS:     2,
	}

	// Parse protobuf wire format to extract SentencePiece pieces.
	// ModelProto.pieces is field 1 (repeated SentencePieceProto).
	// SentencePieceProto has: field 1 = piece (string), field 2 = score (float), field 3 = type (int32)
	id := 0
	i := 0
	for i < len(data) {
		// Read field tag
		tag, n := protoVarint(data[i:])
		if n == 0 {
			break
		}
		i += n

		fieldNum := tag >> 3
		wireType := tag & 7

		if fieldNum == 1 && wireType == 2 {
			// Length-delimited: this is a SentencePieceProto message
			msgLen, n := protoVarint(data[i:])
			if n == 0 {
				break
			}
			i += n

			// Parse the sub-message to get the piece string
			piece := parseSPPiece(data[i : i+int(msgLen)])
			if piece != "" {
				t.Vocab[piece] = id
				t.Inverse[id] = piece
			}
			id++
			i += int(msgLen)
		} else {
			// Skip other fields
			i += protoSkip(data[i:], wireType)
		}
	}

	if len(t.Vocab) == 0 {
		return nil, fmt.Errorf("no vocab entries found in %s", path)
	}

	// Set special tokens
	if id, ok := t.Vocab["<s>"]; ok {
		t.BOS = id
	}
	if id, ok := t.Vocab["</s>"]; ok {
		t.EOS = id
	}

	return t, nil
}

// parseSPPiece extracts the piece string from a SentencePieceProto message.
func parseSPPiece(data []byte) string {
	i := 0
	for i < len(data) {
		tag, n := protoVarint(data[i:])
		if n == 0 {
			break
		}
		i += n

		fieldNum := tag >> 3
		wireType := tag & 7

		if fieldNum == 1 && wireType == 2 {
			// piece string
			strLen, n := protoVarint(data[i:])
			if n == 0 {
				break
			}
			i += n
			if i+int(strLen) <= len(data) {
				return string(data[i : i+int(strLen)])
			}
			return ""
		}
		i += protoSkip(data[i:], wireType)
	}
	return ""
}

// protoVarint reads a varint from data, returns value and bytes consumed.
func protoVarint(data []byte) (uint64, int) {
	var val uint64
	for i := 0; i < len(data) && i < 10; i++ {
		b := data[i]
		val |= uint64(b&0x7f) << (7 * i)
		if b < 0x80 {
			return val, i + 1
		}
	}
	return 0, 0
}

// protoSkip skips a protobuf field value based on wire type.
func protoSkip(data []byte, wireType uint64) int {
	switch wireType {
	case 0: // varint
		for i := 0; i < len(data) && i < 10; i++ {
			if data[i] < 0x80 {
				return i + 1
			}
		}
		return len(data)
	case 1: // 64-bit
		return 8
	case 2: // length-delimited
		l, n := protoVarint(data)
		return n + int(l)
	case 5: // 32-bit
		return 4
	default:
		return len(data) // unknown, consume all
	}
}

// Encode tokenizes text into token IDs.
// Short text uses exact BPE merge rules. Long text (>8KB) switches to
// greedy vocab lookup for speed — the vocab contains all merged tokens.
func (t *Tokenizer) Encode(text string) []int {
	var tokens []int
	if t.BOS >= 0 {
		tokens = append(tokens, t.BOS)
	}

	if len(t.Merges) > 0 {
		if len(text) > 8192 {
			tokens = append(tokens, t.encodeBPEGreedy(text)...)
		} else {
			tokens = append(tokens, t.encodeBPEExact(text)...)
		}
	} else {
		tokens = append(tokens, t.encodeGreedy(text)...)
	}
	return tokens
}

// encodeBPEExact applies merge rules for exact HuggingFace parity.
func (t *Tokenizer) encodeBPEExact(text string) []int {
	words := gptPreTokenize(text)
	var allIDs []int
	for _, word := range words {
		allIDs = append(allIDs, t.mergeWord(word)...)
	}
	return allIDs
}

// mergeWord runs BPE merge rules on a single pretokenized word.
func (t *Tokenizer) mergeWord(word string) []int {
	tokens := make([]string, 0, len(word))
	for i := 0; i < len(word); i++ {
		tokens = append(tokens, byteToHFChar(word[i]))
	}

	for len(tokens) >= 2 {
		bestIdx := -1
		bestPri := len(t.Merges) + 1
		for i := 0; i < len(tokens)-1; i++ {
			key := tokens[i] + " " + tokens[i+1]
			if pri, ok := t.mergePriority[key]; ok && pri < bestPri {
				bestPri = pri
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := tokens[bestIdx] + tokens[bestIdx+1]
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestIdx]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestIdx+2:]...)
		tokens = newTokens
	}

	var ids []int
	for _, tok := range tokens {
		if id, ok := t.Vocab[tok]; ok {
			ids = append(ids, id)
		} else {
			for _, b := range []byte(tok) {
				fallback := fmt.Sprintf("<0x%02X>", b)
				if id, ok := t.Vocab[fallback]; ok {
					ids = append(ids, id)
				}
			}
		}
	}
	return ids
}

// encodeBPEGreedy uses parallel greedy vocab lookup for large inputs.
func (t *Tokenizer) encodeBPEGreedy(text string) []int {
	words := gptPreTokenize(text)

	results := make([][]int, len(words))
	var wg sync.WaitGroup

	const batchSize = 256
	for start := 0; start < len(words); start += batchSize {
		end := start + batchSize
		if end > len(words) {
			end = len(words)
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				results[i] = t.greedyWord(words[i])
			}
		}(start, end)
	}
	wg.Wait()

	var all []int
	for _, ids := range results {
		all = append(all, ids...)
	}
	return all
}

// greedyWord encodes a single word using greedy longest-match on vocab.
func (t *Tokenizer) greedyWord(word string) []int {
	hf := make([]byte, 0, len(word)*3)
	for i := 0; i < len(word); i++ {
		r := byteToUnicode[word[i]]
		hf = append(hf, []byte(string(r))...)
	}
	runes := []rune(string(hf))

	var ids []int
	i := 0
	for i < len(runes) {
		bestLen := 0
		bestID := 0
		maxTry := len(runes) - i
		if maxTry > 32 {
			maxTry = 32
		}
		for end := maxTry; end >= 1; end-- {
			sub := string(runes[i : i+end])
			if id, ok := t.Vocab[sub]; ok {
				bestLen = end
				bestID = id
				break
			}
		}
		if bestLen > 0 {
			ids = append(ids, bestID)
			i += bestLen
		} else {
			b := []byte(string(runes[i]))
			for _, bb := range b {
				fallback := fmt.Sprintf("<0x%02X>", bb)
				if id, ok := t.Vocab[fallback]; ok {
					ids = append(ids, id)
				}
			}
			i++
		}
	}
	return ids
}

// gptPreTokenize splits text into chunks following GPT-2/Qwen conventions:
// - Spaces attach to the following word: "The capital" → ["The", " capital"]
// - Digits are individual tokens: "123" → ["1", "2", "3"]
// - Punctuation groups separately: "hello, world!" → ["hello", ",", " world", "!"]
// - Contractions split: "don't" → ["don", "'t"]
// This replaces the Python regex that Go's regexp engine can't handle (no lookaheads).
// GPTPreTokenize is exported for testing.
func GPTPreTokenize(text string) []string { return gptPreTokenize(text) }

func gptPreTokenize(text string) []string {
	runes := []rune(text)
	var chunks []string
	i := 0
	for i < len(runes) {
		r := runes[i]

		if r == ' ' || r == '\t' {
			// Space: attach to the following word/punct/digit
			start := i
			i++
			if i < len(runes) {
				if unicode.IsLetter(runes[i]) {
					// Space + letters
					for i < len(runes) && unicode.IsLetter(runes[i]) {
						i++
					}
				} else if unicode.IsDigit(runes[i]) {
					// Space + single digit
					i++
				} else if runes[i] != ' ' && runes[i] != '\t' && runes[i] != '\n' && runes[i] != '\r' {
					// Space + punctuation
					for i < len(runes) && !unicode.IsLetter(runes[i]) && !unicode.IsDigit(runes[i]) &&
						runes[i] != ' ' && runes[i] != '\t' && runes[i] != '\n' && runes[i] != '\r' {
						i++
					}
				}
			}
			chunks = append(chunks, string(runes[start:i]))

		} else if r == '\n' || r == '\r' {
			start := i
			for i < len(runes) && (runes[i] == '\n' || runes[i] == '\r') {
				i++
			}
			chunks = append(chunks, string(runes[start:i]))

		} else if unicode.IsLetter(r) {
			// Word
			start := i
			for i < len(runes) && unicode.IsLetter(runes[i]) {
				i++
			}
			// Check for contractions: 's 't 're 've 'm 'll 'd
			if i < len(runes) && runes[i] == '\'' {
				chunks = append(chunks, string(runes[start:i]))
				start = i
				i++ // consume '
				// Grab the contraction suffix
				for i < len(runes) && unicode.IsLetter(runes[i]) {
					i++
				}
				chunks = append(chunks, string(runes[start:i]))
			} else {
				chunks = append(chunks, string(runes[start:i]))
			}

		} else if unicode.IsDigit(r) {
			// Single digit per token
			chunks = append(chunks, string(r))
			i++

		} else {
			// Punctuation / symbols
			start := i
			for i < len(runes) && !unicode.IsLetter(runes[i]) && !unicode.IsDigit(runes[i]) &&
				runes[i] != ' ' && runes[i] != '\t' && runes[i] != '\n' && runes[i] != '\r' {
				i++
			}
			chunks = append(chunks, string(runes[start:i]))
		}
	}
	return chunks
}

// byteToHFChar converts a byte to HuggingFace's byte-level BPE character.
// This is the exact GPT-2 bytes_to_unicode mapping used by HF tokenizers.
var byteToUnicode map[byte]rune
var unicodeToByte map[rune]byte

func init() {
	byteToUnicode = make(map[byte]rune)
	unicodeToByte = make(map[rune]byte)

	// GPT-2 standard mapping: printable bytes map to themselves,
	// non-printable bytes map to U+0100+ range
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
			byteToUnicode[byte(b)] = rune(b)
		} else {
			byteToUnicode[byte(b)] = rune(256 + n)
			n++
		}
	}
	for b, r := range byteToUnicode {
		unicodeToByte[r] = b
	}
}

func byteToHFChar(b byte) string {
	return string(byteToUnicode[b])
}

// ByteToHFCharExported is the exported version of byteToHFChar for tools.
func ByteToHFCharExported(b byte) string {
	return byteToHFChar(b)
}

func (t *Tokenizer) encodeGreedy(text string) []int {
	// Sentencepiece-style: ▁ marks word start, greedy longest match
	normalized := "▁" + strings.ReplaceAll(text, " ", "▁")

	var ids []int
	i := 0
	runes := []rune(normalized)

	for i < len(runes) {
		bestLen := 0
		bestID := 0

		// Try longest match first
		for end := len(runes); end > i; end-- {
			sub := string(runes[i:end])
			if id, ok := t.Vocab[sub]; ok {
				bestLen = end - i
				bestID = id
				break
			}
		}

		if bestLen > 0 {
			ids = append(ids, bestID)
			i += bestLen
		} else {
			// Unknown character — try single-byte fallback
			// Sentencepiece uses <0xXX> for byte fallback
			b := []byte(string(runes[i]))
			for _, bb := range b {
				key := fmt.Sprintf("<0x%02X>", bb)
				if id, ok := t.Vocab[key]; ok {
					ids = append(ids, id)
				}
			}
			i++
		}
	}

	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		if id == t.BOS || id == t.EOS {
			continue
		}
		if word, ok := t.Inverse[id]; ok {
			parts = append(parts, word)
		}
	}
	text := strings.Join(parts, "")

	// If this is a HF BPE tokenizer, reverse the byte mapping
	if len(t.Merges) > 0 {
		var bytes []byte
		for _, r := range text {
			if b, ok := unicodeToByte[r]; ok {
				bytes = append(bytes, b)
			} else {
				// Direct passthrough for regular unicode
				bytes = append(bytes, []byte(string(r))...)
			}
		}
		return string(bytes)
	}

	// Sentencepiece: ▁ → space
	text = strings.ReplaceAll(text, "▁", " ")
	// Only trim leading space if decoding multiple tokens (full sequence).
	// Single-token decode preserves the space for streaming output.
	if len(ids) > 1 {
		text = strings.TrimLeft(text, " ")
	}
	return text
}

// EncodeFile stream-tokenizes a text file to a binary int32 token file.
// Reads line-by-line (constant memory), writes tokens as little-endian int32.
// Returns total token count.
func (t *Tokenizer) EncodeFile(inPath, outPath string) (int64, error) {
	fin, err := os.Open(inPath)
	if err != nil {
		return 0, err
	}
	defer fin.Close()

	fout, err := os.Create(outPath)
	if err != nil {
		return 0, err
	}
	defer fout.Close()

	bw := make([]byte, 0, 16*1024*1024)
	buf := make([]byte, 4)
	var totalTokens int64

	scanner := newLargeScanner(fin)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 {
			continue
		}
		ids := t.Encode(line)
		for _, id := range ids {
			binary.LittleEndian.PutUint32(buf, uint32(int32(id)))
			bw = append(bw, buf...)
		}
		totalTokens += int64(len(ids))
		if len(bw) >= 16*1024*1024 {
			if _, err := fout.Write(bw); err != nil {
				return totalTokens, err
			}
			bw = bw[:0]
		}
	}
	if len(bw) > 0 {
		if _, err := fout.Write(bw); err != nil {
			return totalTokens, err
		}
	}
	if err := scanner.Err(); err != nil {
		return totalTokens, err
	}
	return totalTokens, nil
}

func newLargeScanner(r *os.File) *largeScanner {
	return &largeScanner{r: r, buf: make([]byte, 0, 4*1024*1024)}
}

type largeScanner struct {
	r    *os.File
	buf  []byte
	line string
	err  error
	done bool
	raw  []byte
	pos  int
}

func (s *largeScanner) Scan() bool {
	if s.done {
		return false
	}
	for {
		// Search for newline in remaining raw buffer
		for i := s.pos; i < len(s.raw); i++ {
			if s.raw[i] == '\n' {
				s.line = string(s.raw[s.pos:i])
				s.pos = i + 1
				return true
			}
		}
		// Keep unprocessed tail
		if s.pos < len(s.raw) {
			s.buf = append(s.buf[:0], s.raw[s.pos:]...)
		} else {
			s.buf = s.buf[:0]
		}
		// Read more
		chunk := make([]byte, 4*1024*1024)
		n, err := s.r.Read(chunk)
		if n > 0 {
			s.raw = append(s.buf, chunk[:n]...)
			s.pos = 0
			continue
		}
		if len(s.buf) > 0 {
			s.line = string(s.buf)
			s.buf = s.buf[:0]
			s.done = true
			return true
		}
		if err != nil {
			s.err = err
		}
		s.done = true
		return false
	}
}

func (s *largeScanner) Text() string { return s.line }
func (s *largeScanner) Err() error   { return s.err }

// LoadTokensBin reads a binary int32 token file produced by EncodeFile.
// Uses mmap for constant startup time regardless of file size.
func LoadTokensBin(path string) ([]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	n := len(data) / 4
	tokens := make([]int, n)
	for i := 0; i < n; i++ {
		tokens[i] = int(int32(binary.LittleEndian.Uint32(data[i*4:])))
	}
	return tokens, nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (t *Tokenizer) VocabSize() int {
	return len(t.Vocab)
}

// HasToken returns true if the given string exists in the vocabulary.
func (t *Tokenizer) HasToken(s string) bool {
	_, ok := t.Vocab[s]
	return ok
}

// Keep sort and binary imports used
var _ = sort.Strings
var _ = binary.LittleEndian
var _ = math.Pi
