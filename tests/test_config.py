"""Testes da configuração do estudo de caso."""

from mhwceara.config import CAIXA_CEARA, Caixa, Pastas, raiz_projeto


def test_lon_0360():
    assert CAIXA_CEARA.lon_0360 == (316.0, 326.0)


def test_texto_da_caixa():
    assert str(Caixa(-8.0, 0.0, -44.0, -34.0)) == "8°S a 0°, 44°O a 34°O"


def test_raiz_projeto_sobe_ate_o_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    subpasta = tmp_path / "scripts" / "interna"
    subpasta.mkdir(parents=True)
    assert raiz_projeto(subpasta) == tmp_path.resolve()


def test_pastas():
    pastas = Pastas(raiz_projeto())
    assert pastas.sst.parts[-3:] == ("data", "raw", "sst")
    assert pastas.figuras.parts[-2:] == ("outputs", "figuras")
