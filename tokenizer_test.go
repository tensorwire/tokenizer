package tokenizer

import (
	"testing"
)

func loadGPT2(t *testing.T) *Tokenizer {
	t.Helper()
	tok, err := LoadTokenizer("testdata/gpt2")
	if err != nil {
		t.Fatalf("LoadTokenizer(gpt2): %v", err)
	}
	return tok
}

func TestLoadTokenizer(t *testing.T) {
	tok := loadGPT2(t)
	if tok == nil {
		t.Fatal("tokenizer is nil")
	}
}

func TestVocabSize(t *testing.T) {
	tok := loadGPT2(t)
	size := tok.VocabSize()
	if size < 50000 {
		t.Errorf("vocab size = %d, expected >= 50000 for GPT-2", size)
	}
	t.Logf("GPT-2 vocab size: %d", size)
}

func TestEncodeBasic(t *testing.T) {
	tok := loadGPT2(t)
	ids := tok.Encode("hello world")
	if len(ids) == 0 {
		t.Fatal("Encode returned empty")
	}
	t.Logf("Encode(\"hello world\") = %v (%d tokens)", ids, len(ids))
}

func TestEncodeEmpty(t *testing.T) {
	tok := loadGPT2(t)
	ids := tok.Encode("")
	if len(ids) != 0 {
		t.Errorf("Encode(\"\") = %v, want empty", ids)
	}
}

func TestDecodeRoundTrip(t *testing.T) {
	tok := loadGPT2(t)
	cases := []string{
		"hello world",
		"The quick brown fox jumps over the lazy dog",
		"Hello, World! 123",
		"  spaces  ",
	}
	for _, text := range cases {
		ids := tok.Encode(text)
		decoded := tok.Decode(ids)
		if decoded != text {
			t.Errorf("round-trip failed:\n  input:   %q\n  decoded: %q\n  ids:     %v", text, decoded, ids)
		}
	}
}

func TestGPTPreTokenize(t *testing.T) {
	chunks := GPTPreTokenize("Hello, world! 123")
	if len(chunks) == 0 {
		t.Fatal("GPTPreTokenize returned empty")
	}
	t.Logf("GPTPreTokenize(\"Hello, world! 123\") = %v", chunks)

	joined := ""
	for _, c := range chunks {
		joined += c
	}
	if joined != "Hello, world! 123" {
		t.Errorf("chunks don't reconstruct input: %q", joined)
	}
}

func TestByteToHFChar(t *testing.T) {
	space := ByteToHFCharExported(0x20)
	if space != "Ġ" {
		t.Errorf("byte 0x20 = %q, want Ġ", space)
	}

	excl := ByteToHFCharExported('!')
	if excl != "!" {
		t.Errorf("byte '!' = %q, want !", excl)
	}
}

func TestLoadTokenizer_NotFound(t *testing.T) {
	_, err := LoadTokenizer("/nonexistent/path")
	if err == nil {
		t.Error("LoadTokenizer should fail for nonexistent path")
	}
}

func TestEncodeLong(t *testing.T) {
	tok := loadGPT2(t)
	long := ""
	for i := 0; i < 100; i++ {
		long += "The quick brown fox jumps over the lazy dog. "
	}
	ids := tok.Encode(long)
	if len(ids) == 0 {
		t.Fatal("Encode returned empty for long text")
	}
	decoded := tok.Decode(ids)
	if decoded != long {
		t.Errorf("long text round-trip failed: len(input)=%d, len(decoded)=%d", len(long), len(decoded))
	}
	t.Logf("Long text: %d chars → %d tokens", len(long), len(ids))
}
