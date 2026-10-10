This is "lilia", a QQ chat bot for a small group of friends. It runs next to NapCat (OneBot 11) and answers through an Anthropic-compatible model API. Notes about a particular deployment stay out of the tree: if present, read the untracked `AGENTS.local.md`.

This project is built with C++26 and xmake, on the same toolchain and conventions as liminal. Check the coding style guide at `docs/dev/code-style-guide.md`; the architecture and roadmap are in `docs/dev/design.md`.

Compiler toolchain, C dependencies and python packages are managed by pixi:

- `pixi run configure`: development configure
- `pixi run build`: build the project
- `pixi run format <paths...>`: format given files
- `pixi run test-all`: run all tests (unit + end-to-end)
- `pixi run test-remote-linux --host <ssh-host>` (and `-sanitize`): Linux and ASan/UBSan lanes on a remote host
- `pixi run deploy-cachy [--host <ssh-host>] [--restart]`: build and install a release on the bot host

C++ dependencies should be managed by xmake. Project headers must remain platform-neutral; include Windows SDK and POSIX headers only from implementation units.

`lighter` (coroutine runtime, HTTP, codecs) is an xmake package from `Locietta/lighter`, pinned through the recipe in the `xmake` submodule (`Locietta/xmake-repo`). Change lighter in its own repository; to build lilia against a local lighter checkout, set `LIGHTER_SOURCE_DIR` and run `pixi run xmake require -f -y lighter`. The files copied from liminal's provider layer are listed in `docs/dev/provenance.md`.

Never print or commit secrets: the OneBot access token and model API keys live only in the host's environment file (see `deploy/lilia.service`). Chat logs in the database are private conversations of real people; do not dump them into transcripts beyond what a task needs.

Commit headlines follow `<type>(<scope>): <subject>`.
