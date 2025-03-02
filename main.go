package main

import (
	"bytes"
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"iter"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws/retry"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"golang.org/x/term"
	"rsc.io/markdown"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("llmcli: ")
	if term.IsTerminal(int(os.Stderr.Fd())) {
		log.SetPrefix("\033[1m" + log.Prefix() + ansiReset)
	}
	args := runArgs{model: os.Getenv("LLMCLI_MODEL")}
	flag.StringVar(&args.q, "q", args.q, "your `prompt` to LLM."+
		"\nYou can also provide prompt over stdin."+
		"\nIf you provide data on stdin AND use this flag¹,"+
		"\nthen data provided over stdin are wrapped within <document> tags"+
		"\nand the text provided using this flag goes after that."+
		"\n\n¹ Note that when you use this flag and stdin is a terminal,"+
		"\n it is NOT read to avoid the illusion of blocking.")
	flag.Func("f", "`file` to attach (can be used multiple times)", func(name string) error {
		if name != "" {
			args.attach = append(args.attach, name)
		}
		return nil
	})
	flag.Func("t", "temperature parameter for LLM, [0, 1] range.\nHigher values like 0.8 will make the output more random, while\nlower values like 0.2 will make it more focused and deterministic.", func(val string) error {
		v, err := strconv.ParseFloat(val, 32)
		if err != nil {
			return err
		}
		if v < 0 || v > 1 {
			return errors.New("temperature must be within [0, 1] range")
		}
		x := float32(v)
		args.t = &x
		return nil
	})
	flag.BoolVar(&args.v, "v", args.v, "output some additional details like token usage")
	if configDir, err := os.UserConfigDir(); err == nil {
		args.sys = filepath.Join(configDir, "llmcli", "system-prompt.txt")
	}
	flag.StringVar(&args.sys, "s", args.sys, "system prompt `file`")
	flag.BoolVar(&args.web, "w", args.web, "treat reply as markdown, convert it to html and open result in a browser")
	flag.Func("b", "optional “thinking” budget in `tokens` for Claude 3.7 (≥1024)", func(val string) error {
		if !strings.Contains(args.model, "claude-3-7") {
			return errors.New("“thinking” budget option is for Claude 3.7 model only")
		}
		v, err := strconv.ParseUint(val, 10, 17)
		if err != nil {
			return err
		}
		if v < 1024 {
			return errors.New("thinking tokens budget should be greater than or equal to 1024")
		}
		if v < 4000 {
			log.Print("Anthropic suggests trying at least 4,000 tokens to achieve more comprehensive and nuanced reasoning")
		}
		args.budget = int(v)
		return nil
	})
	flag.Parse()
	if args.q == "" && len(flag.Args()) != 0 {
		args.q = strings.Join(flag.Args(), " ")
	}
	if err := run(context.Background(), args); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && len(ee.Stderr) != 0 {
			os.Stderr.Write(ee.Stderr)
		}
		log.Fatal(err)
	}
}

type runArgs struct {
	model  string // to be set from env only
	q      string
	sys    string
	attach []string
	v      bool
	web    bool
	t      *float32
	budget int
}

func run(ctx context.Context, args runArgs) error {
	if filepath.Base(os.Args[0]) == "chatgpt" {
		return chatgpt(ctx, args)
	}
	prompt, err := readPrompt(args)
	if err != nil {
		return err
	}
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()
	var contentBlocks []types.ContentBlock
	handler := loadHandlers()
	for _, name := range slices.Compact(args.attach) {
		block, err := handler.attToBlock(ctx, name)
		if err != nil {
			return err
		}
		contentBlocks = append(contentBlocks, block)
	}
	contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{Value: prompt})

	cfg, err := config.LoadDefaultConfig(ctx, config.WithSharedConfigProfile("llmcli"))
	var e config.SharedConfigProfileNotExistError
	if errors.As(err, &e) {
		cfg, err = config.LoadDefaultConfig(ctx)
	}
	if err != nil {
		return err
	}
	cl := bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		o.Retryer = retry.NewStandard(func(o *retry.StandardOptions) { o.MaxAttempts = 6 })
	})

	const fallbackModelId = "anthropic.claude-3-sonnet-20240229-v1:0"
	var modelId = cmp.Or(args.model, "anthropic.claude-3-5-sonnet-20240620-v1:0")
	switch modelId {
	case "haiku":
		modelId = "anthropic.claude-3-haiku-20240307-v1:0"
	}
	input := &bedrockruntime.ConverseStreamInput{
		ModelId: &modelId,
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: contentBlocks,
			},
		},
	}
	if args.budget != 0 {
		input.AdditionalModelRequestFields = thinking(args.budget)
		maxTokens := int32(args.budget * 2) // if thinking is enabled, max_tokens must be set and be greater than thinking tokens
		input.InferenceConfig = &types.InferenceConfiguration{MaxTokens: &maxTokens}
	}
	systemPrompt := time.Now().Local().AppendFormat(nil, "Today is Monday, 02 Jan 2006, time zone MST")
	if args.sys != "" {
		if b, err := os.ReadFile(args.sys); err == nil {
			b = bytes.TrimSpace(b)
			if len(b) != 0 && utf8.Valid(b) {
				systemPrompt = append(systemPrompt, ".\n"...)
				systemPrompt = append(systemPrompt, b...)
			}
		}
	}
	input.System = []types.SystemContentBlock{&types.SystemContentBlockMemberText{Value: string(systemPrompt)}}
	if args.t != nil && args.budget != 0 { // “Thinking isn’t compatible with temperature”
		input.InferenceConfig = &types.InferenceConfiguration{Temperature: args.t}
	}
	out, err := cl.ConverseStream(ctx, input)
	var te *types.ThrottlingException
	if errors.As(err, &te) {
		if ok, _ := strconv.ParseBool(os.Getenv("LLMCLI_FALLBACK_ON_THROTTLE")); ok && *input.ModelId != fallbackModelId {
			log.Printf("all retries were throttled, falling back to model %s", fallbackModelId)
			s := fallbackModelId
			input.ModelId = &s
			out, err = cl.ConverseStream(ctx, input)
		}
	}
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	var wr io.Writer = os.Stdout
	if args.web {
		wr = io.MultiWriter(os.Stdout, &buf)
	}
	rc := newResponseConsumer(out)
	if args.budget != 0 {
		// just in case we changed stdout formatting at the start of “thinking”
		// but failed mid-way before resetting formatting
		defer termWrite(ansiReset)
	}
	var thinking bool
	for chunk := range rc.Chunks() {
		if !thinking && chunk.thinking {
			thinking = true
			termWrite(ansiItalic) // bypassing wr
		} else if thinking && !chunk.thinking {
			thinking = false
			termWrite(ansiReset) // bypassing wr
			io.WriteString(wr, "\n\n* * *\n\n")
		}
		io.WriteString(wr, chunk.text)
	}
	if err := rc.Err(); err != nil {
		return err
	}
	if usage := rc.Usage(); args.v && usage != nil {
		log.Printf("tokens usage: total: %d, input: %d, output: %d", *usage.TotalTokens, *usage.InputTokens, *usage.OutputTokens)
	}
	if args.web && buf.Len() != 0 {
		return renderAndOpen(&buf)
	}
	return nil
}

var stdoutIsTerm = sync.OnceValue(func() bool { return term.IsTerminal(int(os.Stdout.Fd())) })

func termWrite(s string) {
	if !stdoutIsTerm() {
		return
	}
	os.Stdout.WriteString(s)
}

func newResponseConsumer(cso *bedrockruntime.ConverseStreamOutput) *responseConsumer {
	return &responseConsumer{cso: cso}
}

type responseConsumer struct {
	cso   *bedrockruntime.ConverseStreamOutput
	usage *types.TokenUsage
	err   error
}

type chunk struct {
	text     string
	thinking bool
}

func (r *responseConsumer) Err() error               { return r.err }
func (r *responseConsumer) Usage() *types.TokenUsage { return r.usage }
func (r *responseConsumer) Chunks() iter.Seq[chunk] {
	return func(yield func(chunk) bool) {
		stream := r.cso.GetStream()
		defer stream.Close()
		for evt := range stream.Events() {
			switch v := evt.(type) {
			case *types.ConverseStreamOutputMemberContentBlockDelta:
				switch d := v.Value.Delta.(type) {
				case *types.ContentBlockDeltaMemberText:
					if !yield(chunk{text: d.Value}) {
						return
					}
				case *types.ContentBlockDeltaMemberReasoningContent:
					if b, ok := d.Value.(*types.ReasoningContentBlockDeltaMemberText); ok && !yield(chunk{text: b.Value, thinking: true}) {
						return
					} else if _, ok := d.Value.(*types.ReasoningContentBlockDeltaMemberRedactedContent); ok && !yield(chunk{text: "\n[…redacted thinking…]\n", thinking: true}) {
						return
					}
				}
			case *types.ConverseStreamOutputMemberContentBlockStop:
			case *types.ConverseStreamOutputMemberMessageStart:
			case *types.ConverseStreamOutputMemberMessageStop:
				if !yield(chunk{text: "\n"}) {
					return
				}
				if s := v.Value.StopReason; s != types.StopReasonEndTurn {
					r.err = fmt.Errorf("stop reason: %s", s)
					return
				}
			case *types.ConverseStreamOutputMemberMetadata:
				r.usage = v.Value.Usage
			default:
				log.Printf("unknown event type %T: %+v", evt, evt)
			}
		}
		r.err = stream.Err()
	}
}

func readPrompt(args runArgs) (string, error) {
	stdinIsTerminal := term.IsTerminal(int(os.Stdin.Fd()))
	var pb strings.Builder
	var stdinData []byte
	var err error
	if stdinIsTerminal && args.q == "" {
		log.Println("Please type your prompt, when done, submit with ^D")
	}
	if !stdinIsTerminal || (stdinIsTerminal && args.q == "") {
		stdinData, err = io.ReadAll(os.Stdin)
	}
	if err != nil {
		return "", err
	}
	if len(bytes.TrimSpace(stdinData)) == 0 && args.q == "" {
		return "", errors.New("empty prompt: please feed it over stdin and/or use the -q flag")
	}
	if !utf8.Valid(stdinData) {
		return "", errors.New("can only take valid utf8 data on stdin")
	}
	switch args.q {
	case "":
		pb.Write(stdinData)
	default:
		if len(bytes.TrimSpace(stdinData)) != 0 {
			pb.WriteString(tagDocOpen)
			pb.Write(stdinData)
			pb.WriteString(tagDocClose)
			pb.WriteByte('\n')
		}
		pb.WriteString(args.q)
	}
	if stdinIsTerminal && args.q == "" {
		log.Println("end of prompt")
	}
	return pb.String(), nil
}

func contentBlockFromFile(p string) (types.ContentBlock, error) {
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	if len(b) > 50<<20 {
		return nil, errors.New("maximum document size supported is 50Mb")
	}
	ct := http.DetectContentType(b)
	if strings.HasPrefix(ct, "image/") {
		block := &types.ContentBlockMemberImage{
			Value: types.ImageBlock{Source: &types.ImageSourceMemberBytes{Value: b}},
		}
		switch ct {
		case "image/jpeg":
			block.Value.Format = types.ImageFormatJpeg
		case "image/png":
			block.Value.Format = types.ImageFormatPng
		case "image/gif":
			block.Value.Format = types.ImageFormatGif
		case "image/webp":
			block.Value.Format = types.ImageFormatWebp
		default:
			return nil, fmt.Errorf("file %s is of unsupported content-type %s", p, ct)
		}
		return block, nil
	}

	docName := strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))
	block := &types.ContentBlockMemberDocument{
		Value: types.DocumentBlock{
			Source: &types.DocumentSourceMemberBytes{Value: b},
			Name:   &docName,
		},
	}
	switch strings.ToLower(filepath.Ext(p)) {
	case ".pdf":
		block.Value.Format = types.DocumentFormatPdf
	case ".md", ".mkd":
		block.Value.Format = types.DocumentFormatMd
	case ".html":
		block.Value.Format = types.DocumentFormatHtml
	case ".doc":
		block.Value.Format = types.DocumentFormatDoc
	case ".docx":
		block.Value.Format = types.DocumentFormatDocx
	case ".csv":
		block.Value.Format = types.DocumentFormatCsv
	case ".txt":
		block.Value.Format = types.DocumentFormatTxt
	default:
		if ct == "text/plain; charset=utf-8" {
			block.Value.Format = types.DocumentFormatTxt
		} else {
			return nil, fmt.Errorf("file %s is of unsupported content-type %s", p, ct)
		}
	}
	// If the attachment looks like a plain text, change it from the attachment
	// block into the text part of the prompt, wrapped within <document> tags.
	// We do this because Claude 3.5 Sonnet only supports image attachments,
	// and if there are attachments of other types, a separate condition in the
	// code downgrades request to use an older Claude 3 Sonnet model.
	// By putting plain text attachments inside the prompt we increase the likelihood
	// of staying within Claude 3.5 Sonnet attachment limits.
	switch block.Value.Format {
	case types.DocumentFormatMd, types.DocumentFormatTxt, types.DocumentFormatCsv:
		if utf8.Valid(b) {
			text := []byte(tagDocOpen[:len(tagDocOpen)-1]) // without the trailing newline
			text = append(text, "<filename>"...)
			text = append(text, p...)
			text = append(text, "</filename>\n"...)
			text = append(text, b...)
			if text[len(text)-1] != '\n' {
				text = append(text, '\n')
			}
			text = append(text, tagDocClose...)
			return &types.ContentBlockMemberText{Value: string(text)}, nil
		}
	}
	return block, nil
}

func loadHandlers() *attHandlers {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return nil
	}
	f, err := os.Open(filepath.Join(configDir, "llmcli", "att-handlers.json"))
	if err != nil {
		return nil
	}
	defer f.Close()
	var matchers []attMatch
	if err := json.NewDecoder(f).Decode(&matchers); err != nil || len(matchers) == 0 {
		return nil
	}
	return &attHandlers{byPrefix: matchers}
}

type attHandlers struct {
	byPrefix []attMatch
}

type attMatch struct {
	Prefix string   `json:"prefix"`
	Cmd    []string `json:"cmd"`
}

func (h *attHandlers) attToBlock(ctx context.Context, name string) (types.ContentBlock, error) {
	if h == nil {
		return contentBlockFromFile(name)
	}
	for _, m := range h.byPrefix {
		if m.Prefix == "" || len(m.Cmd) == 0 || !strings.HasPrefix(name, m.Prefix) {
			continue
		}
		args := slices.Clone(m.Cmd[1:])
		var found bool
		for i := range args {
			if args[i] == "${ARG}" {
				args[i] = name
				found = true
				break
			}
		}
		if !found {
			args = append(args, name)
		}
		cmd := exec.CommandContext(ctx, m.Cmd[0], args...)
		b, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("running %v: %w", cmd, err)
		}
		if !utf8.Valid(b) {
			return nil, fmt.Errorf("command %v output is not a valid utf8", cmd)
		}
		text := []byte(tagDocOpen)
		text = append(text, b...)
		if text[len(text)-1] != '\n' {
			text = append(text, '\n')
		}
		text = append(text, tagDocClose...)
		return &types.ContentBlockMemberText{Value: string(text)}, nil
	}
	return contentBlockFromFile(name)
}

const (
	tagDocOpen  = "<document>\n"
	tagDocClose = "</document>\n"
)

// renderAndOpen converts Markdown content to HTML and opens it in the default browser.
func renderAndOpen(buf *bytes.Buffer) error {
	f, err := os.CreateTemp("", "llmcli_*.html")
	if err != nil {
		return err
	}
	defer f.Close()
	name := f.Name()
	if _, err := f.WriteString(htmlHead); err != nil {
		return err
	}
	p := markdown.Parser{Table: true, AutoLinkText: true}
	body := []byte(htmlHead)
	body = append(body, markdown.ToHTML(p.Parse(buf.String()))...)
	if _, err := f.Write(body); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	var openCmd string
	switch runtime.GOOS {
	case "darwin":
		openCmd = "open"
	case "linux", "freebsd":
		openCmd = "xdg-open"
	case "windows":
		openCmd = "explorer.exe"
	default:
		return fmt.Errorf("don't know how to open %q on %s", name, runtime.GOOS)
	}
	return exec.Command(openCmd, name).Run()
}

//go:embed head.html
var htmlHead string

func thinking(budget int) document.Interface {
	var out struct {
		Thinking struct {
			Type   string `document:"type"`
			Budget int    `document:"budget_tokens"`
		} `document:"thinking"`
	}
	out.Thinking.Type = "enabled"
	out.Thinking.Budget = budget
	return document.NewLazyDocument(out)
}

const ansiReset = "\033[0m"
const ansiItalic = "\033[3m"
