import numpy as np

from nazariya.cluster import EventState, assign_online, from_blob, to_blob

H = 3600.0


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_similar_articles_join_dissimilar_found_new_events():
    vecs = np.stack([unit(1, 0, 0), unit(0.95, 0.1, 0), unit(0, 1, 0), unit(0.9, 0.05, 0.1)])
    labels, sims, state = assign_online(vecs, np.array([0, H, 2 * H, 3 * H]), ["a", "b", "c", "d"], tau=0.8)
    assert labels[0] == labels[1] == labels[3] != labels[2]
    assert len(state) == 2
    assert state[labels[0]].count == 3 and len(state[labels[0]].outlets) == 3
    assert sims[0] == 1.0 and 0.8 <= sims[1] < 1.0


def test_processing_follows_publication_time_not_input_order():
    vecs = np.stack([unit(1, 0.1, 0), unit(1, 0, 0)])
    labels, sims, _ = assign_online(vecs, np.array([5 * H, 0]), ["a", "b"], tau=0.9)
    assert labels == [0, 0]
    assert sims[1] == 1.0  # the earlier article founded the event


def test_events_close_after_window():
    vecs = np.stack([unit(1, 0, 0), unit(1, 0, 0)])
    labels, _, state = assign_online(vecs, np.array([0, 80 * H]), ["a", "b"], tau=0.5, window_s=72 * H)
    assert labels[0] != labels[1] and len(state) == 2


def test_existing_state_is_extended():
    prior = [EventState(id=7, sum=unit(0, 0, 1) * 3, count=3, first_seen=0, last_seen=H)]
    labels, _, state = assign_online(np.stack([unit(0, 0.1, 1)]), np.array([2 * H]), ["x"], tau=0.8,
                                     state=prior, article_ids=[42])
    assert labels == [0] and state[0].id == 7 and state[0].count == 4 and state[0].members == [42]


def test_blob_roundtrip():
    v = unit(0.3, -0.2, 0.9)
    assert np.allclose(from_blob(to_blob(v)), v, atol=1e-3)
    assert np.array_equal(from_blob(to_blob(v, np.float32), np.float32), v)
