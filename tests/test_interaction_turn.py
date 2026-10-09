import numpy as np

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.pipeline.crop import crop_jpeg_base64
from pmai_core.pipeline.interaction_turn import InteractionTurn, choose_person


def _person(gid: str, confianza: float) -> GlobalObjectForContext:
    return GlobalObjectForContext(
        id_global=gid,
        etiqueta=gid,
        confianza=confianza,
        camera_id="cam-a",
        bbox=(0, 0, 2, 2),
    )


def test_choose_person_breaks_ties_by_id():
    chosen = choose_person([_person("b", 0.5), _person("a", 0.5), _person("c", 0.4)])
    assert chosen is not None
    assert chosen.id_global == "a"


def test_turn_without_person_skips_caption_and_speaks():
    calls = {"caption": 0, "speak": [], "execute": []}

    def caption(_: str) -> str:
        calls["caption"] += 1
        return "no"

    turn = InteractionTurn(
        objects=lambda: [],
        frame=lambda camera_id: None,
        caption=caption,
        load_catalog=lambda: [{"id": 1, "name": "Reposo", "arduino_id": 1, "description": ""}],
        answer=lambda obj, catalog: {"respuesta": "Hola", "movimientos": [1, 99]},
        speak=lambda text: calls["speak"].append(text),
        execute=lambda action_id, arduino_id: calls["execute"].append((action_id, arduino_id)),
    )
    reply = turn.ask("¿Quién eres tú ahora?", source="chat")
    assert reply == "Hola"
    assert calls["caption"] == 0
    assert calls["speak"] == ["Hola"]
    assert calls["execute"] == [(1, 1)]


def test_blip_failure_still_answers():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def boom(_: str) -> str:
        raise RuntimeError("blip")

    turn = InteractionTurn(
        objects=lambda: [_person("a", 0.9)],
        frame=lambda camera_id: frame,
        caption=boom,
        load_catalog=lambda: [],
        answer=lambda obj, catalog: {"respuesta": f"vi:{obj.contexto}", "movimientos": []},
        speak=lambda text: None,
        execute=lambda action_id, arduino_id: None,
    )
    assert turn.ask("una pregunta larga de chat", source="chat") == "vi:None"


def test_second_call_while_busy_is_not_spoken_for_chat():
    import threading

    started = threading.Event()
    release = threading.Event()
    spoken: list[str] = []

    def answer(obj, catalog):
        started.set()
        assert release.wait(timeout=2)
        return {"respuesta": "tarde", "movimientos": []}

    turn = InteractionTurn(
        objects=lambda: [],
        frame=lambda camera_id: None,
        caption=lambda image: "",
        load_catalog=lambda: [],
        answer=answer,
        speak=lambda text: spoken.append(text),
        execute=lambda action_id, arduino_id: None,
    )
    worker = threading.Thread(target=lambda: turn.ask("pregunta suficientemente larga", source="chat"))
    worker.start()
    assert started.wait(timeout=2)
    busy = turn.ask("otra pregunta larga", source="chat")
    release.set()
    worker.join(timeout=2)
    assert busy == "Sigo respondiendo, un momento."
    assert "Sigo respondiendo, un momento." not in spoken


def test_short_voice_does_not_call_the_model():
    called = {"answer": 0}
    spoken = []
    turn = InteractionTurn(
        objects=lambda: [],
        frame=lambda camera_id: None,
        caption=lambda image: "",
        load_catalog=lambda: [],
        answer=lambda obj, catalog: called.__setitem__("answer", 1),
        speak=lambda text: spoken.append(text),
        execute=lambda action_id, arduino_id: None,
    )
    reply = turn.ask("hola", source="voice")
    assert reply == "No te escuche con claridad, puedes repetirlo por favor."
    assert spoken == [reply]
    assert called["answer"] == 0


def test_crop_is_raw_jpeg_base64():
    import base64

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)
    encoded = crop_jpeg_base64(frame, (1, 1, 5, 5))
    raw = base64.b64decode(encoded)
    assert raw[:2] == b"\xff\xd8"
    assert not encoded.startswith("data:")
