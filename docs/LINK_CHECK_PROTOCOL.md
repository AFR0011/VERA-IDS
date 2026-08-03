# Link Check Protocol

`tests/test_release_contents.py::test_relative_markdown_links_resolve` validates
every relative Markdown link in the tracked tree. Web links are not fetched by the
test because network state is nondeterministic; dataset links were manually checked
against official provider pages during the 2026-08-03 audit. Re-run the focused
test and manually verify external links before each tagged release.
