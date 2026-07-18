from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

TestClient = pytest.importorskip("fastapi.testclient").TestClient
create_app = pytest.importorskip("hyperoptax.dashboard.api").create_app
models = pytest.importorskip("hyperoptax.dashboard.models")
FilterOperator = models.FilterOperator
SortDirection = models.SortDirection
TrialQuery = models.TrialQuery


class FakeQueryService:
    def __init__(self) -> None:
        self.last_query: TrialQuery | None = None

    def list_studies(self) -> tuple[dict[str, Any], ...]:
        return ({"study_id": "study-1", "name": "Example"},)

    async def describe_study(self, study_id: str) -> dict[str, Any]:
        if study_id != "study-1":
            raise KeyError(study_id)
        return {
            "study": {
                "study_id": study_id,
                "name": "Example",
                "status": "completed",
                "revision": 4,
            },
            "trial_counts": {"completed": 2},
        }

    def query_trials(
        self,
        study_id: str,
        query: TrialQuery,
    ) -> dict[str, Any]:
        if study_id != "study-1":
            raise KeyError(study_id)
        self.last_query = query
        return {
            "rows": [
                {
                    "trial_id": "trial-1",
                    "trial_name": "calm-otter-1",
                    "state": "completed",
                    "objective_value": 0.75,
                    "params": {"learning_rate": 0.01},
                }
            ],
            "rows_total": 1,
            "rows_returned": 1,
            "sampled": False,
            "study_revision": 4,
        }

    def available_fields(self, study_id: str) -> tuple[str, ...]:
        if study_id != "study-1":
            raise KeyError(study_id)
        return ("objective_value", "params.learning_rate", "state", "trial_name")

    def hyperparameter_importance(self, study_id: str) -> dict[str, Any]:
        if study_id != "study-1":
            raise KeyError(study_id)
        return {
            "parameters": [
                {
                    "field": "params.learning_rate",
                    "importance": 0.8,
                    "correlation": 0.6,
                }
            ],
            "n_trials": 8,
            "n_estimators": 200,
            "random_state": 0,
            "study_revision": 4,
            "reason": None,
        }

    def pareto_front(
        self,
        study_id: str,
        *,
        x_field: str,
        y_field: str,
        x_direction: str,
        y_direction: str | None,
    ) -> dict[str, Any]:
        if study_id != "study-1":
            raise KeyError(study_id)
        return {
            "x_field": x_field,
            "y_field": y_field,
            "x_direction": x_direction,
            "y_direction": y_direction or "maximize",
            "points": [{"trial_id": "trial-1", "x": 2.0, "y": 0.75}],
            "trials_considered": 1,
            "study_revision": 4,
        }

    def get_trial(self, study_id: str, trial_id: str) -> dict[str, Any]:
        if study_id != "study-1" or trial_id != "trial-1":
            raise KeyError((study_id, trial_id))
        return {
            "trial_id": trial_id,
            "trial_name": "calm-otter-1",
            "study_id": study_id,
            "objective_value": 0.75,
        }

    def get_changes(self, study_id: str, since_revision: int) -> dict[str, Any]:
        if study_id != "study-1":
            raise KeyError(study_id)
        return {
            "study_revision": 4,
            "study_status": "completed",
            "trials": [],
            "requested_revision": since_revision,
        }


@pytest.fixture
def service() -> FakeQueryService:
    return FakeQueryService()


@pytest.fixture
def client(service: FakeQueryService) -> TestClient:
    return TestClient(create_app(service))


def test_study_and_static_routes(client: TestClient) -> None:
    studies = client.get("/api/v1/studies")
    assert studies.status_code == 200
    assert studies.json() == [{"study_id": "study-1", "name": "Example"}]

    detail = client.get("/api/v1/studies/study-1")
    assert detail.status_code == 200
    assert detail.json()["study"]["revision"] == 4

    index = client.get("/")
    assert index.status_code == 200
    assert "Hyperoptax dashboard" in index.text
    assert "static/plotly.min.js" in index.text
    assert "static/logo-transparent.png" in index.text
    assert "Experiment explorer" in index.text
    assert "Name, ID, state, or value" in index.text
    assert 'id="trial-inspector"' in index.text

    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "api/v1/" in script.text
    assert "https://cdn" not in script.text
    assert "explorerViewSpec" in script.text
    assert "objectiveDurationPareto" in script.text
    assert "function trialName" in script.text
    assert "const IMPORTANCE_SIGNIFICANT_DIGITS = 3" in script.text
    assert "text: displayField(x)" in script.text
    assert "text: displayField(y)" in script.text

    logo = client.get("/static/logo-transparent.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"


def test_structured_query_is_translated_to_models(
    client: TestClient,
    service: FakeQueryService,
) -> None:
    response = client.post(
        "/api/v1/studies/study-1/trials/query",
        json={
            "columns": ["trial_id", "objective_value"],
            "filters": [{"field": "state", "op": "eq", "value": "completed"}],
            "sort": [{"field": "objective_value", "direction": "desc"}],
            "limit": 50,
            "version": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["rows_returned"] == 1
    assert response.json()["rows"][0]["trial_name"] == "calm-otter-1"
    assert service.last_query is not None
    assert service.last_query.columns == ("trial_id", "objective_value")
    assert service.last_query.filters[0].op is FilterOperator.EQ
    assert service.last_query.sort[0].direction is SortDirection.DESC
    assert service.last_query.limit == 50
    assert service.last_query.version == 1


def test_agent_field_discovery_and_trial_detail(client: TestClient) -> None:
    fields = client.get("/api/v1/studies/study-1/fields")
    assert fields.status_code == 200
    assert fields.json() == {
        "fields": [
            "objective_value",
            "params.learning_rate",
            "state",
            "trial_name",
        ],
        "query_version": 1,
    }

    trial = client.get("/api/v1/studies/study-1/trials/trial-1")
    assert trial.status_code == 200
    assert trial.json()["objective_value"] == 0.75
    assert trial.json()["trial_name"] == "calm-otter-1"

    assert client.get("/api/v1/studies/study-1/trials/missing").status_code == 404


def test_changes_and_validation_errors(client: TestClient) -> None:
    changes = client.get(
        "/api/v1/studies/study-1/changes",
        params={"since_revision": 3},
    )
    assert changes.status_code == 200
    assert changes.json()["requested_revision"] == 3

    assert client.get("/api/v1/studies/missing").status_code == 404


def test_hyperparameter_importance_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/studies/study-1/importance")
    assert response.status_code == 200
    assert response.json()["parameters"] == [
        {
            "field": "params.learning_rate",
            "importance": 0.8,
            "correlation": 0.6,
        }
    ]
    assert client.get("/api/v1/studies/missing/importance").status_code == 404
    assert (
        client.get(
            "/api/v1/studies/study-1/changes",
            params={"since_revision": -1},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/studies/study-1/trials/query",
            json={"filters": [{"field": "state", "op": "execute"}]},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/studies/study-1/trials/query",
            json={"version": 2},
        ).status_code
        == 422
    )


def test_direction_aware_pareto_endpoint(client: TestClient) -> None:
    response = client.get(
        "/api/v1/studies/study-1/pareto",
        params={
            "x_field": "duration_seconds",
            "y_field": "objective_value",
            "x_direction": "minimize",
            "y_direction": "maximize",
        },
    )

    assert response.status_code == 200
    assert response.json()["points"] == [{"trial_id": "trial-1", "x": 2.0, "y": 0.75}]
    assert response.json()["y_direction"] == "maximize"
    assert client.get("/api/v1/studies/missing/pareto").status_code == 404
    assert (
        client.get(
            "/api/v1/studies/study-1/pareto",
            params={"x_direction": "sideways"},
        ).status_code
        == 422
    )


def test_base_path_mount(service: FakeQueryService) -> None:
    client = TestClient(create_app(service, base_path="/user/alice/proxy/8080"))

    redirect = client.get("/", follow_redirects=False)
    assert redirect.status_code in {307, 308}
    assert redirect.headers["location"] == "/user/alice/proxy/8080/"

    index = client.get("/user/alice/proxy/8080/")
    assert index.status_code == 200
    studies = client.get("/user/alice/proxy/8080/api/v1/studies")
    assert studies.status_code == 200
