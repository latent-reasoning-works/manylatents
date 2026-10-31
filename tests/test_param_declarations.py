"""Parameter declarations: the format, the generated declarations and their coverage.

The format is defined by the prose in docs/algorithms.md ("Parameter declarations"), by
``manylatents.params.check_declaration`` and by the fixtures in
``tests/fixtures/param_declarations/``.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from manylatents import params
from manylatents.algorithms.latent import get_algorithm, list_algorithms
from manylatents.algorithms.latent.latent_module_base import LatentModule

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "param_declarations"
LIGHTNING_CONFIGS = REPO / "manylatents" / "configs" / "algorithms" / "lightning"
DOMAINS = LIGHTNING_CONFIGS / "domains"

ENTRY_KEYS = {
    "legal", "default", "sensible", "ladder", "limits", "role", "precomputed", "cost",
    "meaning", "more", "less",
}


# ---------------------------------------------------------------------------
# The format: fixtures
# ---------------------------------------------------------------------------

def _fixtures(kind: str) -> list[Path]:
    root = FIXTURES / kind
    assert root.is_dir(), root
    files = sorted(root.glob("*.yaml"))
    assert files, root
    return files


def _expected_message(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("# expect: "):
            return line[len("# expect: "):]
    raise AssertionError(f"{path.name} names no expected message")


@pytest.mark.parametrize("path", _fixtures("valid"), ids=lambda p: p.stem)
def test_valid_fixture_has_no_problem(path):
    doc = yaml.safe_load(path.read_text())
    assert params.check_declaration(doc) == []


def test_one_valid_fixture_per_group():
    groups = sorted(yaml.safe_load(p.read_text())["group"] for p in _fixtures("valid"))
    assert groups == ["latent", "lightning", "tool"]


@pytest.mark.parametrize("path", _fixtures("invalid"), ids=lambda p: p.stem)
def test_invalid_fixture_gives_its_named_message(path):
    doc = yaml.safe_load(path.read_text())
    expected = _expected_message(path)
    problems = params.check_declaration(doc)
    assert any(expected in p for p in problems), (expected, problems)


def test_invalid_fixtures_cover_every_rule():
    names = {p.stem for p in _fixtures("invalid")}
    assert {
        "unknown_key", "missing_key", "bad_type", "default_refused", "sensible_log_lo",
        "limit_fact", "bad_role", "empty_meaning", "bad_cost",
    } <= names


def test_hand_written_yaml_tool_declaration_passes():
    text = (FIXTURES / "valid" / "tool.yaml").read_text()
    doc = yaml.safe_load(text)
    assert doc["origin"] != "manylatents"
    assert params.check_declaration(doc) == []


def _valid_doc() -> dict:
    return yaml.safe_load((FIXTURES / "valid" / "tool.yaml").read_text())


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.update(schema_version=2), "schema_version: 2 is not 1"),
        (lambda d: d.update(group="model"), "group: 'model' is not one of latent, lightning, tool"),
        (lambda d: d.pop("origin"), "missing key 'origin'"),
        (lambda d: d.update(extra=1), "unknown key 'extra'"),
        (lambda d: d["params"]["min_counts"].update(ladder=[100, -1]), "params.min_counts.ladder[1]: -1 is refused by legal"),
        (lambda d: d["params"]["min_counts"].update(
            sensible={"lo": -5, "hi": 10, "scale": "linear", "source": {"kind": "docs", "ref": "x"}}),
         "params.min_counts.sensible.lo: -5 is refused by legal"),
        (lambda d: d["params"]["min_counts"].update(
            sensible={"lo": 10, "hi": 5, "scale": "linear", "source": {"kind": "docs", "ref": "x"}}),
         "params.min_counts.sensible: lo 10 is greater than hi 5"),
        (lambda d: d["params"]["min_counts"].update(
            sensible={"lo": 1, "hi": 5, "scale": "exp", "source": {"kind": "docs", "ref": "x"}}),
         "params.min_counts.sensible.scale: 'exp' is not one of linear, log"),
        (lambda d: d["params"]["min_counts"].update(
            sensible={"lo": 1, "hi": 5, "scale": "linear", "source": {"kind": "guess", "ref": "x"}}),
         "params.min_counts.sensible.source.kind: 'guess' is not one of docs, paper, measured"),
        (lambda d: d["params"]["min_counts"].update(precomputed=True),
         "params.min_counts.precomputed: precomputed is true but role is 'knob', not 'fixed'"),
        (lambda d: d["params"]["min_counts"].update(meaning="x" * 121),
         "params.min_counts.meaning: must be 1 to 120 characters"),
        (lambda d: d["params"]["min_counts"].update(more=""),
         "params.min_counts.more: must be null or 1 to 120 characters"),
        (lambda d: d["params"]["min_counts"].update(cost={"time": "sideways", "memory": "none", "note": None}),
         "params.min_counts.cost.time: 'sideways' is not one of up, down, none"),
        (lambda d: d["params"]["min_counts"].update(cost={"time": "up", "memory": "slow", "note": None}),
         "params.min_counts.cost.memory: 'slow' is not one of up, down, none"),
        (lambda d: d["params"]["min_counts"].update(cost={"time": "up", "memory": "up"}),
         "params.min_counts.cost.note: missing key 'note'"),
        (lambda d: d["params"]["min_counts"].update(cost={"time": "up", "memory": "up", "note": None, "screen": "x"}),
         "params.min_counts.cost.screen: unknown key 'screen'"),
        (lambda d: d["params"]["min_counts"]["limits"].append({"op": "ge", "fact": "n_samples", "offset": 0}),
         "params.min_counts.limits[0].op: 'ge' is not one of le, lt"),
        (lambda d: d["params"]["flavor"].update(default="pearson"),
         "params.flavor.default: 'pearson' is refused by legal"),
    ],
)
def test_check_declaration_refuses(mutate, expected):
    doc = _valid_doc()
    mutate(doc)
    problems = params.check_declaration(doc)
    assert any(expected in p for p in problems), (expected, problems)


def test_check_declaration_reports_every_problem():
    doc = _valid_doc()
    doc["params"]["min_counts"]["role"] = "tunable"
    doc["params"]["flavor"]["meaning"] = ""
    assert len(params.check_declaration(doc)) == 2


def test_every_message_starts_with_a_dotted_field_path():
    import re

    paths = re.compile(r"^[A-Za-z_][\w]*(\.[\w]+|\[\d+\])*: ")
    for path in _fixtures("invalid"):
        for message in params.check_declaration(yaml.safe_load(path.read_text())):
            assert paths.match(message), message


def test_check_declaration_of_a_non_mapping():
    assert params.check_declaration([]) == ["declaration: expected a mapping, got list"]


# ---------------------------------------------------------------------------
# Generated declarations: latent
# ---------------------------------------------------------------------------

def test_phate_declaration():
    doc = params.domain("phate")
    assert doc["origin"] == "manylatents"
    assert doc["group"] == "latent"
    assert doc["method"] == "phate"
    p = doc["params"]
    assert p["t"]["legal"]["type"] == "int"
    assert p["t"]["legal"]["min"] == 1
    assert p["t"]["legal"]["also"] == ["auto"]
    # graphtools replaces knn > n_samples - 2 by n_samples - 2, so only knn <= n_samples - 2
    # is honoured (graphtools 2.1.0 graphs.py, kNNGraph.__init__).
    assert {"op": "le", "fact": "n_samples", "offset": -2} in p["knn"]["limits"]
    assert p["knn"]["role"] == "knob"
    assert p["random_state"]["role"] == "fixed"
    for name in ("n_jobs", "verbose", "device"):
        assert p[name]["role"] == "infra", name


def test_type_and_default_come_from_the_signature():
    from manylatents.algorithms.latent.phate import PHATEModule

    sig = inspect.signature(PHATEModule.__init__)
    doc = params.domain("phate")
    for name, entry in doc["params"].items():
        assert entry["default"] == sig.parameters[name].default, name
    assert doc["params"]["gamma"]["legal"]["type"] == "float"
    assert doc["params"]["knn_dist"]["default"] == "euclidean"


def _latent_classes() -> dict[type, str]:
    classes: dict[type, str] = {}
    for name in list_algorithms():
        classes.setdefault(get_algorithm(name), name)
    return classes


def _source_keys(cls) -> list[str]:
    sig = inspect.signature(cls.__init__)
    return [
        n for n, p in sig.parameters.items()
        if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    ]


@pytest.mark.parametrize("cls", list(_latent_classes()), ids=lambda c: c.__name__)
def test_every_latent_module_declares_every_parameter(cls):
    assert "PARAMS" in cls.__dict__, f"{cls.__name__} declares no PARAMS"
    assert sorted(cls.PARAMS) == sorted(_source_keys(cls)), cls.__name__
    for name, entry in cls.PARAMS.items():
        assert "default" not in entry, f"{cls.__name__}.PARAMS[{name!r}] repeats the default"


@pytest.mark.parametrize("name", params.names("latent"))
def test_latent_declaration_passes_check(name):
    doc = params.domain(name)
    assert params.check_declaration(doc) == []
    assert doc["origin"] == "manylatents"


def test_names_cover_the_registry():
    canonical = {name for cls, name in _latent_classes().items()}
    assert set(params.names("latent")) == canonical
    assert params.names("latent") == sorted(params.names("latent"))


def test_a_module_without_params_fails_and_names_the_class():
    class UndeclaredModule(LatentModule):
        def __init__(self, n_components: int = 2, **kwargs):
            super().__init__(n_components=n_components, **kwargs)

    with pytest.raises(ValueError, match="UndeclaredModule"):
        params._latent_declaration(UndeclaredModule, "undeclared")


def test_a_params_entry_that_repeats_the_default_is_refused():
    class RepeatingModule(LatentModule):
        PARAMS = {"n_components": {"min": 1, "default": 2, "role": "knob", "meaning": "Width."}}  # noqa: RUF012

        def __init__(self, n_components: int = 2, **kwargs):
            super().__init__(n_components=n_components, **kwargs)

    with pytest.raises(ValueError, match="default"):
        params._latent_declaration(RepeatingModule, "repeating")


def test_a_params_type_that_contradicts_the_signature_is_refused():
    class ContradictingModule(LatentModule):
        PARAMS = {"n_components": {"type": "bool", "role": "knob", "meaning": "Width."}}  # noqa: RUF012

        def __init__(self, n_components: int = 2, **kwargs):
            super().__init__(n_components=n_components, **kwargs)

    with pytest.raises(ValueError, match="n_components"):
        params._latent_declaration(ContradictingModule, "contradicting")


def test_a_missing_or_extra_params_entry_is_refused():
    class ShortModule(LatentModule):
        PARAMS = {"k": {"role": "knob", "meaning": "Neighbours."}}  # noqa: RUF012

        def __init__(self, n_components: int = 2, **kwargs):
            super().__init__(n_components=n_components, **kwargs)

    with pytest.raises(ValueError, match="n_components"):
        params._latent_declaration(ShortModule, "short")


def test_unknown_names_raise_key_error():
    with pytest.raises(KeyError):
        params.domain("no_such_method")
    with pytest.raises(KeyError):
        params.domain("no_such_config", group="lightning")
    with pytest.raises(ValueError, match="group"):
        params.domain("phate", group="tool")


def test_latent_config_name_resolves_through_the_config():
    # phate_torchdr is a config, not a registry name; its _target_ is PHATEModule.
    doc = params.domain("phate_torchdr")
    assert doc["method"] == "phate_torchdr"
    assert set(doc["params"]) == set(params.domain("phate")["params"])


# ---------------------------------------------------------------------------
# check reads legal and limits only
# ---------------------------------------------------------------------------

def test_check_accepts_a_value_off_the_ladder():
    assert params.check(params.domain("phate"), {"knn": 7}) == []


def test_check_refuses_a_value_that_a_data_limit_refuses():
    problems = params.check(params.domain("phate"), {"knn": 100}, n_samples=50)
    assert len(problems) == 1
    assert "knn" in problems[0] and "n_samples" in problems[0]


def test_check_skips_a_limit_whose_fact_is_not_given():
    assert params.check(params.domain("phate"), {"knn": 100}) == []


def test_check_refuses_an_infra_parameter_by_role():
    problems = params.check(params.domain("phate"), {"n_jobs": 4})
    assert len(problems) == 1
    assert "n_jobs" in problems[0] and "infra" in problems[0]


def test_check_refuses_a_fixed_parameter_by_role():
    problems = params.check(params.domain("phate"), {"random_state": 1})
    assert len(problems) == 1
    assert "random_state" in problems[0] and "fixed" in problems[0]


def test_check_refuses_illegal_and_unknown_values():
    doc = params.domain("phate")
    assert params.check(doc, {"t": "auto", "gamma": 0.5}) == []
    problems = params.check(doc, {"t": 0, "gamma": 2.0, "knn": 5.5, "nope": 1})
    assert len(problems) == 4
    assert any(p.startswith("t=0") for p in problems)
    assert any("nope" in p for p in problems)


def test_check_does_not_read_sensible_or_ladder():
    doc = _valid_doc()
    doc["params"]["min_counts"]["sensible"] = {
        "lo": 100, "hi": 500, "scale": "linear", "source": {"kind": "measured", "ref": "x"},
    }
    assert params.check(doc, {"min_counts": 7}) == []


def test_check_applies_n_features_limits():
    doc = _valid_doc()
    doc["params"]["n_top_genes"]["role"] = "knob"
    doc["params"]["n_top_genes"]["precomputed"] = False
    assert params.check(doc, {"n_top_genes": 30}, n_features=30) == []
    problems = params.check(doc, {"n_top_genes": 31}, n_features=30)
    assert len(problems) == 1 and "n_features" in problems[0]


# ---------------------------------------------------------------------------
# Generated declarations: Lightning
# ---------------------------------------------------------------------------

def _lightning_config_stems() -> list[str]:
    assert LIGHTNING_CONFIGS.is_dir(), LIGHTNING_CONFIGS
    return sorted(
        p.stem for p in LIGHTNING_CONFIGS.glob("*.yaml")
        if p.stem != "default" and not p.stem.startswith("_")
    )


def test_every_lightning_config_has_a_domain_file():
    assert DOMAINS.is_dir(), DOMAINS
    stems = _lightning_config_stems()
    assert stems
    assert sorted(p.stem for p in DOMAINS.glob("*.yaml")) == stems
    assert params.names("lightning") == stems


@pytest.mark.parametrize("name", _lightning_config_stems())
def test_lightning_declaration_passes_check_and_declares_every_scalar(name):
    doc = params.domain(name, group="lightning")
    assert params.check_declaration(doc) == []
    assert doc["origin"] == "manylatents"
    assert doc["group"] == "lightning"
    composed = params._compose_lightning(name)
    declared = set(doc["params"])
    scalars = set(params._lightning_scalar_paths(composed))
    assert scalars, name
    assert scalars - declared == set(), f"{name}: undeclared {sorted(scalars - declared)}"
    assert declared - scalars == set(), f"{name}: unresolved {sorted(declared - scalars)}"
    for path, entry in doc["params"].items():
        assert entry["default"] == params._lookup(composed, path), path


def test_lightning_domain_files_never_repeat_the_default():
    assert DOMAINS.is_dir(), DOMAINS
    for path in sorted(DOMAINS.glob("*.yaml")):
        source = yaml.safe_load(path.read_text())
        for key, entry in source.items():
            assert "default" not in entry, f"{path.name}: {key}"


def test_cflows_reaches_the_full_chain_without_importing_lightning_modules():
    code = (
        "import json, sys\n"
        "from manylatents import params\n"
        "doc = params.domain('cflows', group='lightning')\n"
        "names = params.names('lightning')\n"
        "for n in names: params.domain(n, group='lightning')\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('manylatents.algorithms.lightning')"
        " or m.startswith('manylatents.lightning') or m == 'lightning' or m.startswith('lightning.'))\n"
        "print(json.dumps({'keys': sorted(doc['params']), 'loaded': loaded}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO),
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    keys = result["keys"]
    for prefix in ("network.", "optimizer.", "trainer.", "loss."):
        assert any(k.startswith(prefix) for k in keys), prefix
    assert result["loaded"] == []


def test_lightning_declaration_needs_no_global_hydra_state():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str((REPO / "manylatents" / "configs").resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        params.domain("ae_reconstruction", group="lightning")
        assert GlobalHydra.instance().is_initialized()
        cfg = compose(config_name="config", overrides=["algorithms/latent=pca"])
        assert cfg.algorithms.latent is not None
    GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def _cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "manylatents.params", *args],
        capture_output=True, text=True, env=env, cwd=str(REPO), stdin=subprocess.DEVNULL, check=False,
    )


def test_show_is_json():
    out = _cli("show", "phate")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == json.loads(json.dumps(params.domain("phate")))


def test_show_lightning_group():
    out = _cli("show", "cflows", "--group", "lightning")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["group"] == "lightning"


def test_show_unknown_name_exits_2():
    out = _cli("show", "no_such_method")
    assert out.returncode == 2
    assert out.stdout == ""


def test_check_names_the_field(tmp_path):
    doc = _valid_doc()
    doc["params"]["knn"] = copy.deepcopy(doc["params"]["min_counts"])
    doc["params"]["knn"]["legal"]["min"] = 1
    doc["params"]["knn"]["default"] = 0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc))
    out = _cli("check", str(bad))
    assert out.returncode == 1, out.stderr
    result = json.loads(out.stdout)
    assert result["ok"] is False
    assert any(p["path"] == "params.knn.default" for p in result["problems"]), result
    assert all(set(p) == {"path", "message"} for p in result["problems"])


def test_check_valid_json_file_exits_0(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_valid_doc()))
    out = _cli("check", str(good))
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {"ok": True, "problems": []}


@pytest.mark.parametrize("content", [None, "{not: [valid", "- just\n- a list\n"])
def test_check_unreadable_file_exits_2(tmp_path, content):
    path = tmp_path / "file.yaml"
    if content is not None:
        path.write_text(content)
    out = _cli("check", str(path))
    if content is not None and content.startswith("-"):
        # readable, but not a mapping: a problem of the document, not of the file
        assert out.returncode == 1
    else:
        assert out.returncode == 2
    result = json.loads(out.stdout)
    assert result["ok"] is False and result["problems"]


def test_dump_writes_one_valid_file_per_method(tmp_path):
    out = _cli("dump", str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert tmp_path.is_dir()
    expected = {f"latent.{n}.json" for n in params.names("latent")}
    expected |= {f"lightning.{n}.json" for n in params.names("lightning")}
    assert {p.name for p in tmp_path.glob("*.json")} == expected
    for path in sorted(tmp_path.glob("*.json")):
        doc = json.loads(path.read_text())
        assert params.check_declaration(doc) == [], path.name
        assert all(set(entry) == ENTRY_KEYS for entry in doc["params"].values())


def test_dump_to_an_unwritable_directory_exits_2(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    out = _cli("dump", str(blocker / "sub"))
    assert out.returncode == 2


# ---------------------------------------------------------------------------
# The distance lists in manylatents.params.sources match the libraries
# ---------------------------------------------------------------------------

def test_isomap_metrics_match_sklearn():
    from sklearn.manifold import Isomap

    from manylatents.params import sources

    options = Isomap._parameter_constraints["metric"][0].options
    assert sorted(sources.ISOMAP_METRICS) == sorted(options)


def test_pdist_metrics_match_scipy():
    from scipy.spatial import distance

    from manylatents.params import sources

    assert sorted(sources.PDIST_METRICS) == sorted(distance._METRICS_NAMES)


def test_umap_metrics_match_umap_learn():
    umap_distances = pytest.importorskip("umap.distances")
    pynn = pytest.importorskip("pynndescent.distances")
    from manylatents.params import sources

    expected = set(umap_distances.named_distances) | set(pynn.named_distances) | {"precomputed"}
    assert sorted(sources.UMAP_METRICS) == sorted(expected)


def test_opentsne_metrics_match_opentsne():
    nn = pytest.importorskip("openTSNE.nearest_neighbors")
    pynn = pytest.importorskip("pynndescent.distances")
    from manylatents.params import sources

    expected = (set(nn.Annoy.VALID_METRICS) | set(nn.NNDescent.VALID_METRICS)
                | set(pynn.named_distances) | {"precomputed"})
    assert sorted(sources.OPENTSNE_METRICS) == sorted(expected)


def test_phate_distances_match_phate():
    phate = pytest.importorskip("phate")
    import ast

    from manylatents.params import sources

    with pytest.raises(ValueError) as excinfo:
        phate.PHATE(knn_dist="no_such_distance")
    listed = ast.literal_eval(str(excinfo.value).split("Choose from ", 1)[1])
    assert sources.PHATE_DISTANCES == listed
