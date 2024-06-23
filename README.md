# llmcli

llmcli is a command-line interface tool for interacting with Large Language Models (LLMs) through AWS Bedrock. It allows users to send prompts and receive responses from AI models directly from the terminal.

## Features

- Send text prompts to LLMs
- Attach files (including images and documents) to your queries
- Read prompts from stdin or command-line arguments
- Stream responses in real-time

## Usage

Basic usage:

```
llmcli -q "Your prompt here"
```

For more detailed information on available options, run:

```
llmcli -h
```

## Requirements

- AWS account with Bedrock access
- Properly configured AWS credentials
