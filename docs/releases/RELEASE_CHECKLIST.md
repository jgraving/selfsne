# Release Checklist

- [ ] Version bump in `pyproject.toml`
- [ ] Changelog update for new version
  - [ ] Add news fragments to `news/` as you work
  - [ ] Run `python -m towncrier build --version <version>` to collect fragments
  - [ ] Verify entry renders for the new version
- [ ] Tests pass
- [ ] Wheel build succeeds (`python -m build`)
- [ ] Git tag created for release
