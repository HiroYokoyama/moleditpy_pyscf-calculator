from concurrent.futures import ThreadPoolExecutor

from audit_helpers import load_module


def test_concurrent_jobs_claim_distinct_directories(monkeypatch, tmp_path):
    mod = load_module(monkeypatch, "worker")
    workers = [mod.PySCFWorker("", {"out_dir": str(tmp_path)}) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda w: w._make_job_dir(), workers))
    assert len(set(paths)) == len(workers)
    assert len(list(tmp_path.iterdir())) == len(workers)
