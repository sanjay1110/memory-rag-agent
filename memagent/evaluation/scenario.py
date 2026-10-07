"""The scripted three-session scenario used by both the demo and the memory /
continuity evaluation.

Priya migrates the ``orders-api`` service from the legacy v1 platform to
Nimbus over three separate sessions.  Each session is run by a *fresh*
``MemoryAgent`` instance that only shares the on-disk stores with the previous
one, so anything the agent "remembers" genuinely came through persistence.

A turn is either a user message (str) or an agent action (tuple), which lets
the script exercise the explicit task API alongside natural-language progress
tracking.
"""

USER = "priya"

MIGRATION_STEPS = [
    "Containerise the service",
    "Write the service manifest",
    "Move secrets to Vault",
    "Migrate the database",
    "Set up observability",
    "Canary launch",
    "Decommission v1",
]

SESSIONS = [
    [
        "Hi, my name is Priya. I'm migrating the orders-api service from v1 to Nimbus.",
        ("start_task", "Migrate orders-api to Nimbus", MIGRATION_STEPS),
        "I prefer Python examples. Please keep answers concise.",
        "Our region is eu-west-1 because we store EU customer data.",
        "Which base image should I use to containerise it?",
        "I've finished containerise the service.",
        "We decided to use logical replication for the database move.",
    ],
    [
        "Hey, where did we leave off?",
        "Done with the service manifest.",
        "Where do I put the database password for orders-api?",
        "How often do those credentials have to be rotated?",
        "Finished moving secrets to Vault.",
        "How do I move the data without downtime?",
        "I prefer Go examples from now on. The deadline is November 14.",
    ],
    [
        "What do you remember about me and this project?",
        "What did we decide for the database move?",
        "Finished the database migration.",
        "Which alerts do I need before taking production traffic?",
        "How long until I can turn off the v1 machines?",
        "What's the next step?",
    ],
]

# (probe query, substring that must appear in a top-3 memory, substring that must NOT appear)
MEMORY_PROBES = [
    ("What is the user's name?", "Priya", None),
    ("Which region does the user deploy to?", "eu-west-1", None),
    ("Which programming language does the user prefer for examples?", "Go", "Python"),
    ("What did we decide about the database migration?", "logical replication", None),
    ("What is the project deadline?", "November 14", None),
    ("Which service is the user migrating?", "orders-api", None),
    ("How should answers be formatted for this user?", "concise", None),
]

# Facts the scenario *should* have produced (for memory precision).
EXPECTED_FACT_SUBSTRINGS = ["Priya", "orders-api", "Python", "Go", "concise", "eu-west-1",
                            "logical replication", "November 14", "Migrate orders-api"]

# Continuity checks: (session index, turn index, substring expected in the response or context)
CONTINUITY_CHECKS = [
    (1, 0, "response", "Write the service manifest", "Session 2 resumes at the right next step"),
    (1, 0, "context", "Containerise the service", "Completed step from session 1 is visible in session 2"),
    (1, 0, "context", "eu-west-1", "Region fact from session 1 is loaded upfront in session 2"),
    (2, 0, "context", "Go", "Superseded preference (Go) is what session 3 sees"),
    (2, 1, "response", "logical replication", "Decision from session 1 is recalled in session 3"),
    (2, 5, "response", "Set up observability", "Session 3 knows the next step after three sessions of progress"),
]
