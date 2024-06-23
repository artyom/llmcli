package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func main() {
	log.SetFlags(0)
	args := runArgs{}
	flag.StringVar(&args.q, "q", args.q, "your prompt to LLM")
	flag.Func("f", "file to attach (can be used multiple times)", func(name string) error {
		if name != "" {
			args.attach = append(args.attach, name)
		}
		return nil
	})
	flag.Parse()
	if err := run(context.Background(), args); err != nil {
		log.Fatal(err)
	}
}

type runArgs struct {
	q      string
	attach []string
}

func run(ctx context.Context, args runArgs) error {
	if args.q == "" {
		return errors.New("no query provided")
	}
	var contentBlocks []types.ContentBlock
	for _, name := range slices.Compact(args.attach) {
		block, err := contentBlockFromFile(name)
		if err != nil {
			return err
		}
		contentBlocks = append(contentBlocks, block)
	}
	contentBlocks = append(contentBlocks, &types.ContentBlockMemberText{Value: args.q})

	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return err
	}
	cl := bedrockruntime.NewFromConfig(cfg)

	const claudeModelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
	var modelId = claudeModelId
	for i := range contentBlocks {
		if _, ok := contentBlocks[i].(*types.ContentBlockMemberDocument); ok {
			// https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
			// Anthropic Claude 3.5 on AWS Bedrock only supports image attachments, not documents,
			// as of 2024-06-23. If the document is attached, pick another model.
			// https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
			const oldClaudeModelId = "anthropic.claude-3-sonnet-20240229-v1:0"
			log.Printf("Model %s doesn't support documents, only images, falling back to %s model instead.", modelId, oldClaudeModelId)
			modelId = oldClaudeModelId
			break
		}
	}

	out, err := cl.ConverseStream(ctx, &bedrockruntime.ConverseStreamInput{
		ModelId: &modelId,
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: contentBlocks,
			},
		},
	})
	if err != nil {
		return err
	}
	stream := out.GetStream()
	defer stream.Close()
	for evt := range stream.Events() {
		switch v := evt.(type) {
		case *types.ConverseStreamOutputMemberContentBlockDelta:
			if d, ok := v.Value.Delta.(*types.ContentBlockDeltaMemberText); ok {
				fmt.Print(d.Value)
			}
		case *types.ConverseStreamOutputMemberContentBlockStop:
			fmt.Println()
		case *types.ConverseStreamOutputMemberMessageStart:
		case *types.ConverseStreamOutputMemberMessageStop:
		case *types.ConverseStreamOutputMemberMetadata:
		default:
			log.Printf("unknown event type %T: %+v", evt, evt)
		}
	}
	return stream.Err()
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
	switch filepath.Ext(p) {
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
		return nil, fmt.Errorf("file %s is of unsupported content-type %s", p, ct)
	}
	return block, nil
}
