# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial setup
- Licencing
- `pycountry` for ISO standard country codes

### Fixed
- `end_date` now defaults to inifinity
- `start_date` now defaults to today's date
- Extra docs for affected area
- `active_crisis_config` should now import correctly; note untested on server, but works on local Airflow deploy
- `slug` added to `ActiveCrisisConfig`
- `PROJECT_ROOT` properly param'd

[Unreleased]: https://github.com/Flowminder/flowbot-pipeline-root/compare/06572a96a58dc510037d5efa622f9bec8519bc1beab13c9f251e97e657a9d4ed...master
