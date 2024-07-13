package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws/retry"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("llmcli: ")
	if st, err := os.Stderr.Stat(); err == nil && st.Mode()&os.ModeCharDevice != 0 {
		log.SetPrefix("\033[1m" + log.Prefix() + "\033[0m")
	}
	args := runArgs{}
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
	flag.BoolVar(&args.v, "v", args.v, "output some additional details like token usage")
	if configDir, err := os.UserConfigDir(); err == nil {
		args.sys = filepath.Join(configDir, "llmcli", "system-prompt.txt")
	}
	flag.StringVar(&args.sys, "s", args.sys, "system prompt `file`")
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
	q      string
	sys    string
	attach []string
	v      bool
}

func run(ctx context.Context, args runArgs) error {
	var stdinIsTerminal bool
	if st, err := os.Stdin.Stat(); err == nil {
		stdinIsTerminal = st.Mode()&os.ModeCharDevice != 0
	}
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
		return err
	}
	if len(bytes.TrimSpace(stdinData)) == 0 && args.q == "" {
		return errors.New("empty prompt: please feed it over stdin and/or use the -q flag")
	}
	if !utf8.Valid(stdinData) {
		return errors.New("can only take valid utf8 data on stdin")
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
	contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{Value: pb.String()})

	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return err
	}
	cl := bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		o.Retryer = retry.NewStandard(func(o *retry.StandardOptions) { o.MaxAttempts = 6 })
	})

	const oldClaudeModelId = "anthropic.claude-3-sonnet-20240229-v1:0"
	const claudeModelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
	var modelId = claudeModelId
	for i := range contentBlocks {
		if doc, ok := contentBlocks[i].(*types.ContentBlockMemberDocument); ok {
			// https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
			// Anthropic Claude 3.5 on AWS Bedrock only supports image attachments, not documents,
			// as of 2024-06-23. If the document is attached, pick another model.
			// https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
			log.Printf("Model %s doesn't support documents, only images, falling back to %s model instead.", modelId, oldClaudeModelId)
			switch doc.Value.Format {
			case types.DocumentFormatMd, types.DocumentFormatHtml, types.DocumentFormatCsv, types.DocumentFormatTxt:
				log.Print("You can also feed text documents as part of the prompt by ingesting them over stdin.")
			}
			modelId = oldClaudeModelId
			break
		}
	}
	input := &bedrockruntime.ConverseStreamInput{
		ModelId: &modelId,
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: contentBlocks,
			},
		},
		System: []types.SystemContentBlock{&types.SystemContentBlockMemberText{
			Value: time.Now().Local().Format("Today is Monday, 02 Jan 2006, time zone MST")}},
	}
	if args.sys != "" {
		if b, err := os.ReadFile(args.sys); err == nil {
			b = bytes.TrimSpace(b)
			if len(b) != 0 && utf8.Valid(b) {
				input.System = append(input.System, &types.SystemContentBlockMemberText{Value: string(b)})
			}
		}
	}
	out, err := cl.ConverseStream(ctx, input)
	var te *types.ThrottlingException
	if errors.As(err, &te) {
		if ok, _ := strconv.ParseBool(os.Getenv("LLMCLI_FALLBACK_ON_THROTTLE")); ok && *input.ModelId != oldClaudeModelId {
			log.Printf("all retries were throttled, falling back to model %s", oldClaudeModelId)
			s := oldClaudeModelId
			input.ModelId = &s
			out, err = cl.ConverseStream(ctx, input)
		}
	}
	if err != nil {
		return err
	}
	var usage *types.TokenUsage
	stream := out.GetStream()
	defer stream.Close()
	var stopReasonErr error
	for evt := range stream.Events() {
		switch v := evt.(type) {
		case *types.ConverseStreamOutputMemberContentBlockDelta:
			if d, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberText); ok {
				if _, err := os.Stdout.WriteString(d.Value); err != nil {
					return err
				}
			}
		case *types.ConverseStreamOutputMemberContentBlockStop:
		case *types.ConverseStreamOutputMemberMessageStart:
		case *types.ConverseStreamOutputMemberMessageStop:
			if _, err := os.Stdout.WriteString("\n"); err != nil {
				return err
			}
			if s := v.Value.StopReason; s != types.StopReasonEndTurn {
				stopReasonErr = fmt.Errorf("stop reason: %s", s)
			}
		case *types.ConverseStreamOutputMemberMetadata:
			usage = v.Value.Usage
		default:
			log.Printf("unknown event type %T: %+v", evt, evt)
		}
	}
	if err := stream.Close(); err != nil {
		return err
	}
	if err := stream.Err(); err != nil {
		return err
	}
	if args.v && usage != nil {
		log.Printf("tokens usage: total: %d, input: %d, output: %d", *usage.TotalTokens, *usage.InputTokens, *usage.OutputTokens)
	}
	return stopReasonErr
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
			text = append(text, filepath.Base(p)...)
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
		args := append([]string{}, m.Cmd[1:]...)
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
