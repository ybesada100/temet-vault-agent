# systemd templates

Drop these in `~/.config/systemd/user/`, edit the paths, and enable.

## Long-running daemon (preferred)

The Python scheduler handles the cadence itself.

```bash
mkdir -p ~/.config/systemd/user
cp temet-agent.service ~/.config/systemd/user/

# Edit WorkingDirectory + OBSIDIAN_VAULT_PATH inside the file first.
$EDITOR ~/.config/systemd/user/temet-agent.service

systemctl --user daemon-reload
systemctl --user enable --now temet-agent.service
journalctl --user -u temet-agent -f
```

## One-shot timer (alternative)

If you'd rather let systemd own the schedule, change the `.service`'s
`ExecStart` to `... --max-cycles 1` and enable the timer instead:

```bash
cp temet-agent.{service,timer} ~/.config/systemd/user/
$EDITOR ~/.config/systemd/user/temet-agent.service        # add --max-cycles 1
systemctl --user daemon-reload
systemctl --user enable --now temet-agent.timer
systemctl --user list-timers temet-agent.timer
```

## Persistence note

`Persistent=true` in the timer means missed runs (laptop suspended) fire
once on resume. For the long-running variant, `Restart=on-failure` brings
the process back if it crashes.
