package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
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
	req := claudeRequest{
		Messages: []claudeMessage{{Content: args.q}},
	}
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}

	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return err
	}
	cl := bedrockruntime.NewFromConfig(cfg)

	const claudeModelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
	out, err := cl.InvokeModelWithResponseStream(ctx, &bedrockruntime.InvokeModelWithResponseStreamInput{
		Body:        body,
		ModelId:     aws.String(claudeModelId),
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return err
	}
	stream := out.GetStream()
	defer stream.Close()
	for evt := range stream.Events() {
		switch v := evt.(type) {
		case *types.ResponseStreamMemberChunk:
			var msg anthropicStreamEvent
			if err := json.Unmarshal(v.Value.Bytes, &msg); err != nil {
				return err
			}
			if msg.Type == "content_block_delta" && msg.Delta.Type == "text_delta" && msg.Delta.Text != "" {
				fmt.Print(msg.Delta.Text)
			} else if msg.Type == "content_block_stop" {
				fmt.Println()
			}
		default:
			log.Printf("unknown event type %T: %+v", evt, evt)
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}
	return nil
}

type anthropicStreamEvent struct {
	Type  string `json:"type"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

type claudeRequest struct {
	MaxTokens int
	Messages  []claudeMessage
}

func (r claudeRequest) MarshalJSON() ([]byte, error) {
	if len(r.Messages) == 0 {
		return nil, errors.New("request must have at least one message")
	}
	if r.Messages[0].Role != roleUser {
		return nil, errors.New("role of the first message must be user")
	}
	out := struct {
		AnthropicVersion string          `json:"anthropic_version"`
		MaxTokens        int             `json:"max_tokens"`
		Messages         []claudeMessage `json:"messages"`
	}{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        cmp.Or(r.MaxTokens, 1024),
		Messages:         r.Messages,
	}
	return json.Marshal(&out)
}

// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#claude-messages-supported-models
type claudeResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		TokensInput  int `json:"input_tokens"`
		TokensOutput int `json:"output_tokens"`
	} `json:"usage"`
}

// https://docs.anthropic.com/en/api/messages
type claudeMessage struct {
	Role    role   `json:"role"`
	Content string `json:"content"`
}

type role byte

func (r role) MarshalText() ([]byte, error) { return []byte(r.String()), nil }

const (
	roleUser      role = iota // user
	roleAssistant             // assistant
)

//go:generate stringer -type role -linecomment
