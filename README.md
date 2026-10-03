# lilia

A QQ group chat companion for a small group of friends. lilia talks to a NapCat
(OneBot 11) instance over a loopback websocket, keeps a local SQLite log of the
chats it is allowed in, and answers through an Anthropic-compatible model
endpoint (DeepSeek by default) when someone @-mentions it, replies to it or
calls it by name.

Built with C++26 on the same toolchain as [liminal](https://github.com/Locietta/liminal):
GCC 16.2 from Pixi, xmake, and liminal's `lighter` runtime (vendored with
`git subtree`, see [docs/dev/provenance.md](docs/dev/provenance.md)).

## Build and test

```powershell
pixi run configure     # releasedbg
pixi run build
pixi run test-all      # unit tests + end-to-end test (fake NapCat + fake model)
pixi run test-remote-linux --host cachy            # same on Linux
pixi run test-remote-linux-sanitize --host cachy   # ASan/UBSan
```

## Run

```text
lilia [--config <path>] [--check-config] | --version | --help
```

The default config path is `~/.config/lilia/config.json` (`%APPDATA%\lilia\config.json`
on Windows). Start from [deploy/config.example.json](deploy/config.example.json).
Secrets are written as `"$NAME"` and read from the environment; a literal secret
is only accepted when the file is private to its owner. Data (the database and
an optional `persona.md`) lives in `~/.local/share/lilia` unless `data_dir` says
otherwise.

In allowed groups, admins can send `/lilia status` and `/lilia reload` (reloads
`persona.md`).

## Deploy

`pixi run deploy-cachy` uploads the committed worktree to `~/services/lilia/src`
on the host, builds a release there, installs `~/services/lilia/bin/lilia-<commit>`,
points `bin/current` at it and installs [deploy/lilia.service](deploy/lilia.service)
as a systemd user unit. Add `--restart` to restart the running service.
