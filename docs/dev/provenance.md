# Code taken from liminal

## `lighter/` (git subtree)

Vendored on 2026-10-03 from liminal `main` at `d520ce6` (split commit `b7c1208`):

```powershell
# in a throwaway clone of liminal, so the real repository is untouched
git subtree split --prefix=lighter main -b lighter-export
# in lilia
git subtree add --squash --prefix=lighter <clone> lighter-export
```

Treat `lighter/` as read-only. To pick up changes, repeat the split on a fresh
clone and run `git subtree pull --squash --prefix=lighter <clone> lighter-export`.
Changes needed by lilia go into liminal first.

Candidates to upstream into lighter:

- `lilia/net/websocket.*` and `lilia/net/codec.*`: a websocket client over
  `lighter::Tcp`, plus base64 and SHA-1.
- Awaiting a `Task<T>` whose error channel is `void` copies `T`
  (`Outcome<T, void, void>::value()` returns by value for lvalues), so a
  move-only `T` does not compile. Tests work around it with `shared_ptr`.

Plan: move lighter into its own repository once lilia needs its first change to
it, then consume it from both projects as a submodule or an xmake package.

## Provider layer (copied, renamed `liminal::` to `lilia::`)

From liminal `main` at `d520ce6`, keeping the relative paths so fixes can be
ported in either direction:

- `liminal/error.h`, `liminal/text.h`
- `liminal/tools/outcome.{h,cpp}`
- `liminal/provider/{anthropic,auth,compact}.{h,cpp}`, `common.h`, `history.h`, `provider.h`
- `liminal/provider/detail/anthropic_protocol.h`, `completion_retry.{h,cpp}`

The SQLite statement helpers in `lilia/store/sqlite.*` were extracted from the
file-local helpers in `liminal/session/store.cpp`. The build scaffolding
(root `xmake.lua`, `pixi.toml`, formatting, hooks and remote test scripts) and
`lilia/xmake/rules/version.lua` were adapted from liminal.
