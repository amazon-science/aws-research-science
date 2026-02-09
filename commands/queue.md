# Queue Command

Show the experiment job queue status.

When the user invokes this command, show them the current queue status:
- Running jobs (which GPU, PID, when started)
- Queued jobs (waiting for GPU, retry count if any)
- Queue watcher daemon status

Execute: `./ds-exp-plugin/scripts/queue_status.sh`

Present the output to the user cleanly.
