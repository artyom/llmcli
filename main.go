package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func main() {
	log.SetFlags(0)
	args := runArgs{}
	flag.StringVar(&args.q, "q", args.q, "your prompt to LLM")
	flag.Parse()
	if err := run(context.Background(), args); err != nil {
		log.Fatal(err)
	}
}

type runArgs struct {
	q string
}

func run(ctx context.Context, args runArgs) error {
	if args.q == "" {
		return errors.New("no query provided")
	}
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return err
	}
	cl := bedrockruntime.NewFromConfig(cfg)

	const claudeModelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
	var modelId = claudeModelId
	out, err := cl.ConverseStream(ctx, &bedrockruntime.ConverseStreamInput{
		ModelId: &modelId,
		Messages: []types.Message{
			{
				Role:    types.ConversationRoleUser,
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: args.q}},
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
