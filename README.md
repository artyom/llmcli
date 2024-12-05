# llmcli

llmcli is a command-line interface tool for interacting with Large Language Models (LLMs) through AWS Bedrock. It allows users to send prompts and receive responses from AI models directly from the terminal.

## Usage

Basic usage:

```
llmcli "Your prompt here"
```

For more detailed information on available options, run:

```
llmcli -h
```

## Requirements

- AWS account with Bedrock access and at least one [model that supports ConverseStream API and system prompt](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html) enabled.
  Configure which model to use with `LLMCLI_MODEL` environment variable (example: `us.amazon.nova-micro-v1:0`).
- Properly configured AWS credentials.
  This tool tries to use AWS profile named “llmcli”, and falls back to default AWS credentials if profile is not found.

## Examples

Passing input via stdin:

```
git diff | llmcli "Write a commit message for this change"
```

Attaching files (`-f` can be used multiple times to attach more than one file):

```
llmcli -f document.mkd "Please give me a summary of this document"
```

Analyzing images:

```
llmcli -f image.jpg "Describe what you see in this image. Your response would be used verbatim as an 'alt' element text for this image."
```

## Advanced features

This tool allows preprocessing of attachments using external tools, enabling basic customization of attachment handling.
The main use case for this is to integrate it with tools that fetch remote resources.

For example, if you have a tool named `url-to-text`, that takes page url as an argument and outputs plain text from this page on its stdout, you can then call it like so:

```
url-to-text https://en.wikipedia.org/wiki/Linux | llmcli "When did the first version of Linux come out?"
```

If you find yourself doing that kind of calls often, you can configure a callback command instead:

Create a file in the [config directory](https://pkg.go.dev/os#UserConfigDir) at `llmcli/att-handlers.json` path with the following content:

```json
[
    {"prefix":"https://", "cmd":["url-to-text", "${ARG}"]}
]
```

This file configures llmcli to invoke `url-to-text` program to handle “attachments” that start with `https://` prefix, passing that url as the first positional argument to the program (the `${ARG}` placeholder value).

Then you can pass the url as if it was an attachment to `-f` flag:

```
llmcli -f https://en.wikipedia.org/wiki/Linux "When did the first version of Linux come out?"
```

This call would be equivalent to an earlier example, but llmcli would take care of calling `url-to-text` program itself.


## Contributing

While I appreciate interest in this project, please note that I'm not actively seeking outside contributions at this time.
This tool was developed for a specific use case, and I aim to keep its scope focused.
You're welcome to fork the repository for your own needs.
Thank you for understanding.

## License

ISC License, see LICENSE.txt
