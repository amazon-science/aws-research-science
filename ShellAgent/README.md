## This folder contains work in progress, mainly

1. supplementary material for submitted artifacts
2. Example notebook implementations for our Shell agent





## Getting started

#### Install prereqs


In a notebook terminal:
```
apt update

apt install ripgrep pandoc poppler-utils ffmpeg

apt-get install pdfgrep
```

test with:
`pdfgrep`

or

`rga` in terminal, which should print a help command.

If `rga` command is not found, you may have to download the latest release for your system and include it in your path from [here](https://github.com/phiresky/ripgrep-all/releases/tag/v0.10.6). For example:

```
wget https://github.com/phiresky/ripgrep-all/releases/download/v0.10.6/ripgrep_all-v0.10.6-x86_64-unknown-linux-musl.tar.gz

tar -xvf ripgrep_all-v0.10.6-x86_64-unknown-linux-musl.tar.gz 

export PATH=$PATH:ripgrep_all-v0.10.6-x86_64-unknown-linux-musl
```

Also, if this still doesn't work try adding to path from the notebook using this magic command:
```
%env PATH=/root/ripgrep_all-v0.10.6-x86_64-unknown-linux-musl/:$PATH
```

> Precompiled binaries should work most of the time for rga. For pdfgrep, use apt or yum installs


#### Next steps

- Make sure you have the pdfmetadata.sh file next to the main notebook before testing
- Make sure you have a "files" folder next to the main notebook before testing. Place any PDF files or other docs in this folder.
- other installs are in the notebook (like langchain_aws and llamaindex_cli)
- Run python code to do this in the terminal (more stable)

```
python shell_agent.python
```

- This creates an 'output.jsonl' file locally, which can be evaluated using the `shell_agent_eval.ipynb` notebook.

### Benchmark Results

TBD
