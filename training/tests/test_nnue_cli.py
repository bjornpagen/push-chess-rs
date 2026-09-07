"""One NNUE command namespace; routing tests do not train or open a corpus."""
from pathlib import Path

import pytest

from pushzero import cli, nnue


def test_nnue_train_and_export_share_one_namespace(monkeypatch, tmp_path, capsys):
    calls = []
    def train(*args, **kwargs):
        calls.append(("train",args,kwargs))
        return {"ok":True}
    def export(*args):
        calls.append(("export",args))
        return {"ok":True}
    monkeypatch.setattr(nnue,"train",train)
    monkeypatch.setattr(nnue,"export",export)
    cli.main(["--allow-cpu","nnue","train","--db",str(tmp_path),"--runs","2","3",
              "--output",str(tmp_path/"candidate.safetensors"),"--steps","1","--device","CPU"])
    assert calls[0][0] == "train" and calls[0][2]["runs"] == [2,3]
    assert calls[0][2]["device"] == "CPU"
    cli.main(["--allow-cpu","nnue","export","candidate.safetensors","--output",str(tmp_path/"candidate.bin")])
    assert calls[1] == ("export",(Path("candidate.safetensors"),tmp_path/"candidate.bin"))
    assert not list(tmp_path.iterdir())
    assert '"ok": true' in capsys.readouterr().out


def test_no_legacy_nnue_command_alias():
    for argv in (["nnue-export"],["nnue"]):
        with pytest.raises(SystemExit) as error: cli.main(argv)
        assert error.value.code == 2


def test_benchmark_interleaves_and_has_no_saved_work(monkeypatch):
    from pushzero import nnue_benchmark
    calls = []
    arms = {name: lambda item,name=name: calls.append((name,item)) for name in ("a","b")}
    result = nnue_benchmark._measure(arms,[0,1],5)
    assert calls[:8] == [("a",0),("b",0)] * 4
    assert calls[8:12] == [("a",0),("b",0),("b",1),("a",1)]
    assert set(result) == {"a","b"}
    for size,repeats in (([],5),([0],5),([4097],5),([1],1)):
        with pytest.raises(ValueError): nnue_benchmark.compare(size,repeats)
    monkeypatch.setattr(nnue_benchmark,"compare",lambda sizes,repeats: {"sizes":sizes,"repeats":repeats})
    cli.main(["--allow-cpu","nnue","benchmark","--batches","1","256","--repeats","5"])
