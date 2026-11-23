# Changelog fragments

Include a Markdown fragment in each pull request with a user-facing change.
Towncrier combines the fragments into [CHANGELOG.md](../CHANGELOG.md) during
release preparation.

Name each file `<issue>.<category>.md` using the GitHub issue number, for
example `123.fixed.md`. For work without an issue, use a readable name with a
short random suffix and a leading `+`, for example
`+sphere-bounds-a1b2c3d4.fixed.md`. Towncrier leaves these entries unlinked.
For multiple entries in the same category, add a counter before `.md`, for
example `123.fixed.1.md`.

The categories, in release order, are `added`, `changed`, `deprecated`,
`removed`, and `fixed`. Write one entry per file in imperative present tense
and end it with a period. Towncrier adds the Markdown bullet. Include migration
guidance for changes, deprecations, and removals when it applies.

For a change without user-facing impact, you can record a one-line reason in
an optional `.skip` file, such as `+test-cleanup-a1b2c3d4.skip`.

Create an issue-linked fragment with Towncrier or write the file directly.

```console
uvx --from towncrier==25.8.0 towncrier create 123.fixed.md --content "Fix sphere bounds for motion blur."
```

Preview the pending fragments from the repository root. Replace `VERSION`
and `YYYY-MM-DD` with the intended release version and date.

```console
uvx --from towncrier==25.8.0 towncrier build --draft --version VERSION --date YYYY-MM-DD
```

The draft leaves the changelog and fragments intact. CI runs the same preview.
See the [publishing guide](../docs/publishing.md#publish-the-next-version) for
release assembly.
