## GitTutor Bot

A narrow-domain chatbot that answers Git CLI questions .

Deployed on Google Cloud Run and evaluated via an automated harness with deterministic checks and MaaJ LLM-judge evals.



## Live URL

\- \*\*App\*\*:



---



## Topic 



In-scope: Git commands and concepts: staging/index/working tree, commit history, branching, remotes, merge/rebase, conflicts, submodules, common Git errors.



Out-of-scope categories (positive framing):

1\) General programming not related to Git

2\) General knowledge / trivia

3\) Requests involving unauthorized access or security exploitation (redirects to safe alternatives)



### Note: Uncertainty handling (escape hatch)

If the question lacks details, the bot asks for:

- exact `git ...` command

- full output/error text

- user goal



---


## How to run locally? (Powershell Command)

uv sync
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-project-ieor4576"
export GOOGLE_CLOUD_REGION="us-central1"
uv run uvicorn app:app --host 0.0.0.0 --port 8080


