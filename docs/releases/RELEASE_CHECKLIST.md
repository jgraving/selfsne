# Release Checklist

- [ ] Version bump in `pyproject.toml`
- [ ] Changelog update for new version
  - [ ] Automation: run `scripts/update_changelog.py` or use `towncrier` to collect news fragments
  - [ ] Verify entry renders for the new version
- [ ] Tests pass
- [ ] Wheel build succeeds (`python -m build`)
- [ ] Git tag created for release
