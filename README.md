# Flowbot
A Cookiecutter template for Flowbot repos. This is for instantiating and maintaining reporting data pipelines

## Quick start: instantiating a new active crisis
Make sure you have a `.geojson` of your affected areas before you begin - it's helpful to have the path to it in your clipboard.
To instantiate:
1. Clone this repo
   ```bash
   git clone https://github.com/Flowminder/flowbot-pipeline-root.git
   cd flowbot-pipeline-root
1. Install deps with `pipenv install`
1. Run Cookiecutter with `bash run.sh . -o ..`
1. Choose 'active crisis' for the first option
1. Follow the prompts onscreen, pasting the path to the geojson when asked. Some prompts are not needed for the active crisis 
1. Copy the result to the relevant server's `dags` or `flowbot` folder; you should see the new pipeline in Airflow's interface
1. Copy the replay file from `~/.cookiecutter_replays/{your_project_slug}` to this folder; this will support merging changes from the original root repo into your rendered repo.
1. Optional but strongly advised: run `git init` to create a git repo for tracking changes to the new pipeline

## Available pipelines
A pipeline is a combination of a country and a report configiuration, used to render a new repository from the contents of `{{cookiecutter.__project_slug}}`. The rendering process populates the relevant `*_config.py` file and removes any irrelevant notebooks and dags.

### Active crisis
A pipeline for monitoring an upcoming or ongoing crisis. To define, you need:
- The geographic area of the crisis; this is currently a polygon in EPSG:4326 
- A key start date to begin monitoring
- An optional end date

This pipeline produces a daily active crisis report showing total and newly displaced subscribers from affected areas to host locations. For an example report, see <https://www.flowminder.org/resources/publications-reports/haiti-gang-violence-in-downtown-port-au-prince-mobility-situation-report-29-february-12-march-2024>

### Mobility insights
A pipeline for producing monthly reports on national mobility trends.
To define this pipeline, you need:
- A reference date range - this will be a period that builds up a baseline for the rest of the reports to refer to.
- A start date


For more information, see <https://haiti.mobility-dashboard.org/files/HTI%20platform%20release%20documentation%20Nov%202024.pdf>

## Other contents
Some key other parts of the repo:
### flowbot_dataclasses
A set of modules contain the specific configurations for a country and a report type. They are part of the pipenv used in rendering the pipelines.
- CountryStaticData defines a dataclass for country-server specific configurations; for example, if there is a query to fetch the administrative geometry of a country from `flowdb`, it would live in here. If you are implementing existing pipelines on a new in-country server, start here.
- ProjectStructure specifies which parts of `{{cookiecutter._project_slug}}` are kept for a given pipeline, and which `*_config.py` is to be rendered. If you are implementing a new pipeline, start here.
### hooks
The `cookiecutter` hooks for populating the repo.
- `pre_gen_project` is some simple validation rules
- `post_get_project` is the machinery that implements the logic defined in `flowbot_dataclasses`
### local_extensions
Defines a set of custom Jinja filters for accessing `flowbot_dataclasses`

## Populated repo contents

### *_config.py
This is the only Python file populated by the Cookiecutter template (the only other file populated is the README). This is imported into the DAG - if you need to make changes to an existing pipeline, this is where you should start looking. 

### dags
Contains the DAGs associated with this pipeline. Although all piplines currently consist of a single DAG, this may change in the future and is not an assumption you can rely on. If you are extending or modifying a DAG, it is recommended that you add any configuration variable (dates, static file names, region lists, ect) to the pipeline's `config.py`.

### notebooks
Contains the parameterised notebooks and associated helper libraries that form the bulk of the pipeline processing via `flowpyter-task`. These should be as generic as possible and take any date ranges, static data sources or similar from the `config.py` file via a `params` cell. For further information, see <https://github.com/Flowminder/flowpyter-task>

### static
Contains artefacts that are unchanging between dagruns, but may be changed by users. The current two use cases for this are static data files that contain population estimates and the Jinja templates used for reports.

### manual
Room for information that must be added manually per dagrun. The main case for this is the key observations, but it is also used to provide references to previous reports for the back matter.

### data and executed_notebooks
These folders are populated by `flowpyter-task` tasks - they exist, but should be left empty.

## Development
There is a simple approvaltest in the .circleci dir. If you make a change to the structure of the repo, run
```
pipenv run cookiecutter . --replay-file .circleci/test-replay.json -o .circleci/approved -f
```
...to update the approval test. If you make changes to the cookiecutter parameters, you'll need to generate a new replay file; run cookciecutter with your new parameters and copy the replay file to `.circleci/test-replay.json` as above.
