package main

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/artyom/retry"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const openaiTokenEnv = "OPENAI_API_KEY"

func chatgpt(ctx context.Context, args runArgs) error {
	token := os.Getenv(openaiTokenEnv)
	if token == "" {
		return errors.New(openaiTokenEnv + " must be set")
	}

	prompt, err := readPrompt(args)
	if err != nil {
		return err
	}
	var systemPrompt []byte
	if args.sys != "" {
		if b, err := os.ReadFile(args.sys); err == nil {
			b = bytes.TrimSpace(b)
			if len(b) != 0 && utf8.Valid(b) {
				systemPrompt = b
			}
		}
	}
	systemPrompt = time.Now().Local().AppendFormat(systemPrompt, "\nToday is Monday, 02 Jan 2006, time zone MST.")
	systemPrompt = bytes.TrimSpace(systemPrompt)

	userMessage := message{Role: "user"}

	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()
	handler := loadHandlers()
	for _, name := range slices.Compact(args.attach) {
		block, err := handler.attToBlock(ctx, name)
		if err != nil {
			return err
		}
		switch b := block.(type) {
		case *types.ContentBlockMemberText:
			userMessage.Content = append(userMessage.Content, textBlock(b.Value))
		case *types.ContentBlockMemberImage:
			userMessage.Content = append(userMessage.Content, imageBlock(b.Value.Source.(*types.ImageSourceMemberBytes).Value))
		default:
			return fmt.Errorf("file %s is of unsupported type", name)
		}
	}
	userMessage.Content = append(userMessage.Content, textBlock(prompt))

	modelRequest := chatgptRequest{
		Model:  cmp.Or(os.Getenv("LLMCLI_CHATGPT_MODEL"), "gpt-4o"),
		Stream: true,
		Messages: []message{
			{Role: "system", Content: []contentEntry{textBlock(systemPrompt)}},
			userMessage,
		},
		Temperature: args.t,
	}
	payload, err := json.Marshal(modelRequest)
	if err != nil {
		return err
	}
	var userAgent string
	if bi, ok := debug.ReadBuildInfo(); ok {
		userAgent = fmt.Sprintf("%s/%s", bi.Main.Path, bi.Main.Version)
	}
	fn := func() (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+token)
		if userAgent != "" {
			req.Header.Set("User-Agent", userAgent)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}
		defer resp.Body.Close()
		statusErr := &unexpectedStatusError{code: resp.StatusCode}
		if resp.Header.Get("Content-Type") == "application/json" {
			buf := make([]byte, 1024)
			n, _ := io.ReadFull(resp.Body, buf)
			if buf = buf[:n]; len(buf) != 0 {
				statusErr.text = string(buf)
			}
		}
		return nil, statusErr
	}
	rcfg := retry.Config{MaxAttempts: 3, RetryOn: func(err error) bool {
		var e *unexpectedStatusError
		return errors.As(err, &e) && e.code == http.StatusTooManyRequests
	}}
	rcfg = rcfg.WithDelayFunc(func(i int) time.Duration { return time.Second * time.Duration(i) })
	resp, err := retry.FuncVal(ctx, rcfg, fn)
	if err != nil {
		return err
	}
	ct := resp.Header.Get("Content-Type")
	if ct == "text/event-stream; charset=utf-8" {
		return streamResponse(resp.Body)
	}
	if ct != "application/json" {
		return fmt.Errorf("unexpected content-type: %q", ct)
	}
	var out chatgptResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	if l := len(out.Choices); l != 1 {
		return fmt.Errorf("response returned %d choices instead of expected 1", l)
	}
	fmt.Println(out.Choices[0].Message.Content)
	if reason := out.Choices[0].FinishReason; reason != "stop" {
		return fmt.Errorf("stop reason: %s", reason)
	}
	return nil
}

func streamResponse(r io.Reader) error {
	// https://platform.openai.com/docs/api-reference/chat/object
	// https://platform.openai.com/docs/api-reference/streaming
	type chunk struct {
		Otype   string `json:"object"`
		Choices []struct {
			Delta struct {
				Content string  `json:"content"`
				Reason  *string `json:"finish_reason"`
			} `json:"delta"`
		} `json:"choices"`
	}
	w := bufio.NewWriterSize(os.Stdout, 40)
	defer w.Flush()
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		const dataPrefix = "data: "
		const doneChunk = "data: [DONE]"
		b := sc.Bytes()
		if !bytes.HasPrefix(b, []byte(dataPrefix)) {
			continue
		}
		if len(b) == len(doneChunk) && string(b) == doneChunk {
			w.WriteByte('\n')
			break
		}
		var msg chunk
		if err := json.Unmarshal(b[len(dataPrefix):], &msg); err != nil {
			return err
		}
		if msg.Otype != "chat.completion.chunk" || len(msg.Choices) == 0 {
			continue
		}
		w.WriteString(msg.Choices[0].Delta.Content)
		if reason := msg.Choices[0].Delta.Reason; reason != nil && *reason != "stop" {
			return fmt.Errorf("stop reason: %s", *reason)
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return w.Flush()
}

type chatgptResponse struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

type chatgptRequest struct {
	Model       string    `json:"model"`
	Stream      bool      `json:"stream"`
	Messages    []message `json:"messages"`
	Temperature *float32  `json:"temperature,omitempty"`
}

type message struct {
	Role    string         `json:"role"`
	Content []contentEntry `json:"content"`
}

func (m *message) MarshalJSON() ([]byte, error) {
	if len(m.Content) == 1 {
		if text, ok := m.Content[0].(textBlock); ok {
			return json.Marshal(struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			}{Role: m.Role, Content: string(text)})
		}
	}
	type tmp message
	return json.Marshal(tmp(*m))
}

type contentEntry interface {
	MarshalJSON() ([]byte, error)
}

type textBlock string

func (t textBlock) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{Type: "text", Text: string(t)})
}

type imageBlock []byte

func (img imageBlock) MarshalJSON() ([]byte, error) {
	var out []byte
	out = append(out, `{"type":"image_url","image_url":{"url":"data:`...)
	ct := http.DetectContentType(img)
	if !strings.HasPrefix(ct, "image/") {
		return nil, fmt.Errorf("detected non-image content type for imageBlock: %s", ct)
	}
	out = append(out, ct...)
	out = append(out, `;base64,`...)
	out = base64.StdEncoding.AppendEncode(out, img)
	out = append(out, `"}}`...)
	if !json.Valid(out) {
		panic("produced invalid json")
	}
	return out, nil
}

type unexpectedStatusError struct {
	code int
	text string
}

func (e *unexpectedStatusError) Error() string {
	if e.text == "" {
		return fmt.Sprintf("unexpected status: %v", e.code)
	}
	return fmt.Sprintf("unexpected status: %v\n%s", e.code, e.text)
}
