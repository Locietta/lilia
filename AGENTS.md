This is "lilia", a QQ chat bot for a small group of friends. It runs on the `cachy` Linux server next to NapCat (OneBot 11) and answers through an Anthropic-compatible model API. Operational notes for that host live in the user's workspace `D:\.config\wireguard-minecraft` (`deployment/lilia/`, `deployment/napcat/`).

This project is built with C++26 and xmake, on the same toolchain and conventions as liminal. Check the coding style guide at `docs/dev/code-style-guide.md`; the architecture and roadmap are in `docs/dev/design.md`.

Compiler toolchain, C dependencies and python packages are managed by pixi:

- `pixi run configure`: development configure
- `pixi run build`: build the project
- `pixi run format <paths...>`: format given files
- `pixi run test-all`: run all tests (unit + end-to-end)
- `pixi run test-remote-linux --host cachy` (and `-sanitize`): Linux and ASan/UBSan lanes on cachy
- `pixi run deploy-cachy [--restart]`: build and install a release on cachy

C++ dependencies should be managed by xmake. Project headers must remain platform-neutral; include Windows SDK and POSIX headers only from implementation units.

`lighter/` is vendored from liminal with `git subtree` and is read-only here: change it in liminal first, then `git subtree pull`. The files copied from liminal's provider layer are listed in `docs/dev/provenance.md`.

Never print or commit secrets: the OneBot access token and model API keys live only in `~/.config/lilia/secrets.env` on the host. Chat logs in the database are private conversations of real people; do not dump them into transcripts beyond what a task needs.

Commit headlines follow `<type>(<scope>): <subject>`.
