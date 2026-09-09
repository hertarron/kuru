# Upstream merges

kuru is a fork of [janhq/jan](https://github.com/janhq/jan). This document gives the policy for
merges from the upstream `dev` branch, and the procedure to do one.

## Why a policy is necessary

Upstream and kuru want different things. Jan is a general assistant and a coding agent. kuru is a
roleplay client. Some upstream work is a direct gain for kuru. Some of it removes the parts that
kuru is built on.

The second kind is the dangerous kind. Git does not report it as a conflict. When upstream deletes
a file or a block that kuru never touched, the merge applies that deletion without a question. The
feature disappears and the build still passes.

## What kuru takes and what kuru leaves

| Area | Policy | Reason |
|---|---|---|
| `web-app/` chat, threads, messages, transport | Take, then re-apply kuru's changes | Upstream fixes real bugs here, and kuru's roleplay layer sits on top of it |
| `web-app/` Hub, model cards, settings pages | Take with care | kuru redesigned parts of these; a merge can revert the redesign |
| `core/`, shared types, i18n plumbing | Take | Common ground with no kuru opinion |
| `src-tauri/plugins/tauri-plugin-llamacpp` | Leave. kuru owns this | See "The llama.cpp split" |
| `extensions/llamacpp-extension` | Leave. kuru owns this | Same |
| Cowork, agent tools, skills, artifacts | Take, but do not wire in | Coding-agent features. They are inert and they keep the merge small |
| Analytics and telemetry | Never take | kuru removed all of it on purpose |
| Branding (`productName`, locale strings) | Never take | kuru is not Jan |

## The llama.cpp split

Upstream `dev` deleted the llama.cpp downloader and replaced it with an engine that links llama.cpp
into the app. The pin is in `src-tauri/plugins/tauri-plugin-llamacpp/build.rs`. The CI job
`engine-build.yml` compiles it on self-hosted CUDA, ROCm, and Metal runners.

kuru cannot use this. kuru has no runner pool, and a compile-time pin means the user cannot update
llama.cpp without a full native rebuild. kuru keeps the downloader instead.

The cost is permanent. This plugin conflicts on every merge. Resolve it as "ours" every time. Do
not read those conflicts one by one.

Files that kuru owns here, and that upstream has deleted:

- `src-tauri/plugins/tauri-plugin-llamacpp/src/backend.rs`
- `src-tauri/plugins/tauri-plugin-llamacpp/src/router.rs`
- `src-tauri/plugins/tauri-plugin-llamacpp/src/load_probe.rs`
- `src-tauri/plugins/tauri-plugin-llamacpp/src/device.rs`
- `src-tauri/plugins/tauri-plugin-llamacpp/src/deps_analyzer.rs`
- `src-tauri/plugins/tauri-plugin-llamacpp/src/path.rs`
- `extensions/llamacpp-extension/src/backend.ts`
- `web-app/src/hooks/useBackendUpdater.ts`
- `web-app/src/containers/BackendUpdateHistory.tsx`
- `web-app/src/containers/dialogs/BackendUpdater.tsx`

Restore every one of them after a merge.

## When to merge

Do not merge on a schedule. Upstream has published no release since kuru forked, so a version
number is not a signal.

Merge when one of these is true:

- Upstream fixed a bug that kuru has.
- Upstream added something kuru wants, and the commit is easy to find.
- The distance is more than about 300 commits. A larger gap costs more than the merge saves.

Do not merge in the middle of a kuru feature. Finish the feature and commit it first.

## Procedure

1. Commit all work. A conflict can erase an uncommitted file, and git holds no copy of it.
2. Push the commit to `hertarron/kuru`.
3. Run `git fetch origin dev`.
4. Run `git checkout -b merge/upstream-dev`.
5. Run `git merge origin/dev --no-commit`.
6. Resolve every conflict under `src-tauri/plugins/tauri-plugin-llamacpp` and
   `extensions/llamacpp-extension` as "ours". Use `git checkout --ours -- <path>`.
7. Restore the deleted files that the list above names. Use `git checkout HEAD -- <path>`.
8. Resolve the `web-app` conflicts by hand. Read each one.
9. Do the audit of silent losses. The next section gives it.
10. Regenerate `web-app/src/routeTree.gen.ts`.
11. Run the typecheck: `cd web-app` and then `& ".\node_modules\.bin\tsc.cmd" -b`.
12. Run `cargo check` in `src-tauri`.
13. Run the tests: `npx vitest --run` in `web-app`.
14. Start the app and look at it. A merge can pass every check and still break the UI.

## The audit of silent losses

This step finds what the merge removed without a conflict. Do it every time.

List the files that the merge deleted:

```
git diff --name-status HEAD -- web-app extensions src-tauri | grep '^D'
```

Read every line. For each file, decide whether kuru needs it.

Then read the unused-variable errors from the typecheck. Each one is a signal. An import that
kuru used before the merge, and that nothing uses after it, means the merge deleted the code that
called it. Find what it called and put the code back.

In the merge of 2026-09-09 this audit found four losses that no conflict reported:

- The whole backend updater, deleted from `__root.tsx` and from `hooks/`.
- The MTP backend-version gate in `ModelSetting.tsx`.
- `web-app/src/lib/mtp.ts`, which three kuru files import.
- The `mtp.needsUpgrade` locale string.

## Renames

Upstream renames kuru's own work sometimes. `isMtpQuant` became `isSpecSidecar`. `MtpPanel` became
`SpecDraftPanel`. When the upstream version does more than kuru's version, take the upstream name
and delete kuru's copy. Then re-apply the kuru behavior that the rename dropped.

## Commit identity

Every kuru commit and push uses `hertarron <hertarron@users.noreply.github.com>`. Pass it per
command:

```
git -c user.name=hertarron -c user.email=hertarron@users.noreply.github.com commit
```
