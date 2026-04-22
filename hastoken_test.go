package tokenizer

import (
	"testing"
)

func TestHasToken(t *testing.T) {
	tok, err := LoadTokenizer("testdata/gpt2")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	// GPT-2 should have common tokens
	if !tok.HasToken("the") {
		t.Error("expected 'the' in vocab")
	}
	// GPT-2 uses byte-level BPE, space is Ġ
	if !tok.HasToken("Ġthe") && !tok.HasToken(" the") {
		t.Log("note: ' the' not directly in vocab (byte-level BPE)")
	}

	// Should not have random strings
	if tok.HasToken("xyzzy_not_a_token") {
		t.Error("unexpected token found")
	}
}

func TestHasToken_SpecialTokens(t *testing.T) {
	tok, err := LoadTokenizer("testdata/gpt2")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	// GPT-2 doesn't have ChatML tokens
	if tok.HasToken("<|im_start|>") {
		t.Error("GPT-2 should not have <|im_start|>")
	}
	if tok.HasToken("<|im_end|>") {
		t.Error("GPT-2 should not have <|im_end|>")
	}
}
