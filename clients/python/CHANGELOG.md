# Changelog

All notable changes to the `systemone` package. Versions follow [SemVer](https://semver.org); until
1.0, minor versions may change the API.

## 0.1.0 - unreleased

First release, published as `systemone-client` (`system-one` is an unrelated project); the import
name is `systemone`.

- `Client` and `AsyncClient` for any server speaking the System One wire format
  (`POST /v1/systemone`): Kenning, Clef, TypeSafe Jev.
- Question types `Noul`, `Choice`, `Score`; typed answers `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`.
- `Kenning.from_pretrained` (extra `[local]`): run Kenning in-process from a Hugging Face repo id or
  an exported bundle, calibration included.
- Typed (`py.typed`).
