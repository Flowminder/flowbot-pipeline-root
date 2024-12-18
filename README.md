# Flowbot pipeline template

A Cookiecutter template for Flowbot repos.
Make sure you have a `.geojson` of your affected areas before you begin - it's helpful to have the path to it in your clipboard.
To instantiate:
- Clone this repo
- Install deps with `pipenv install`
- Run Cookiecutter with `pipenv run cookiecutter .`
- Answer the questions

It's recommended that you copy the replay file from `~/.cookiecutter_replays/{your_project_slug}` to this folder; this will support merging changes from the original root repo into your rendered repo.

## Dev note; appovaltest
There is a simple approvaltest in the .circleci dir. If you make a change to the structure of the repo, run
```
pipenv run cookiecutter . --replay-file .circleci/test-replay.json -o .circleci/approved -f
```
...to update the approval test. If you make changes to the cookiecutter parameters, you'll need to generate a new replay file; run cookciecutter with your new parameters and copy the replay file to `.circleci/test-replay.json` as above.
