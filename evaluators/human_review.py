"""Utilities for managing LangSmith annotation queues and
adding experiment runs for human evaluation.
"""

from langsmith import Client

QUEUE_NAME = "Target Discovery Response Quality"


def get_client() -> Client:
    return Client()


def get_or_create_queue():
    """
    Retrieve the annotation queue if it exists.
    Otherwise create it.

    Returns
    -------
    AnnotationQueue
        LangSmith annotation queue object.
    """
    try:
        client = get_client()
        queue = client.read_annotation_queue(
            queue_name=QUEUE_NAME
        )

        print(
            f"Found existing annotation queue: "
            f"{QUEUE_NAME}"
        )

        return queue

    except Exception:
        print(
            f"Creating annotation queue: "
            f"{QUEUE_NAME}"
        )

        client = get_client()
        queue = client.create_annotation_queue(
            name=QUEUE_NAME,
            description=(
                "Human review queue for biochemical "
                "target discovery agent evaluations."
            ),
        )

        return queue


def add_experiment_to_queue(experiment_name: str):
    """
    Add all runs from a LangSmith experiment
    to the annotation queue.

    Parameters
    ----------
    experiment_name : str
        LangSmith project / experiment name.
    """
    queue = get_or_create_queue()
    client = get_client()

    runs = list(
        client.list_runs(
            project_name=experiment_name
        )
    )

    run_ids = [run.id for run in runs]

    if not run_ids:
        print(
            f"No runs found for experiment: "
            f"{experiment_name}"
        )
        return

    client.add_runs_to_annotation_queue(
        queue_id=queue.id,
        run_ids=run_ids,
    )

    print(
        f"Added {len(run_ids)} runs to "
        f"annotation queue '{QUEUE_NAME}'."
    )


def add_run_to_queue(run_id: str):
    """
    Add a single run directly to the annotation queue.

    Parameters
    ----------
    run_id : str
        LangSmith run ID.
    """
    queue = get_or_create_queue()
    client = get_client()

    client.add_runs_to_annotation_queue(
        queue_id=queue.id,
        run_ids=[run_id],
    )

    print(
        f"Added run {run_id} to "
        f"annotation queue '{QUEUE_NAME}'."
    )
