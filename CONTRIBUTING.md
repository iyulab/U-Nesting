# Contributing

Thanks for taking the time. This is a small project, so the process is short.

## Reporting a bug

Open an issue with a **reproduction** — a failing test, a short program, or the
exact input and the output you got. A reproduction is worth more than a careful
description, because it removes the guessing from the diagnosis.

Include the version you are on and the platform. If the behaviour differs
between platforms, say so; that is often the whole bug.

**Security problems do not go here** — see [SECURITY.md](SECURITY.md).

## Proposing a change

Open an issue before writing code for anything larger than a fix. The useful
thing to describe is the **problem**, not the patch: what you were trying to do,
what got in the way, and why the existing API could not express it. That leaves
room for a better answer than either of us had in mind at the start.

Additions are weighed against whether they belong in a general-purpose library
at all. A concept that only makes sense inside one caller's domain is better
kept in that caller — a library that absorbs its callers' vocabulary stops
being reusable by anyone else.

## Sending a patch

- Match the surrounding code — its naming, its error handling, its comment density.
- Add a test that fails without the change.
- Run the project's test suite, formatter and linter before pushing.
- Keep the commit message about the observable behaviour that changed, not about
  the circumstances in which you found it.

Public API changes need a note in `CHANGELOG.md` under `[Unreleased]`.
