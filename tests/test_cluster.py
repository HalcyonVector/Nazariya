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



def test_place_rule_keeps_different_states_apart():
    import numpy as np
    from nazariya.cluster import assign_online
    a = np.array([1.0, 0.0, 0.0]); b = np.array([0.95, 0.312, 0.0]); b /= np.linalg.norm(b)  # cosine 0.95
    c = np.array([0.75, 0.66, 0.0]); c /= np.linalg.norm(c)                                   # cosine ~0.75 to a
    vecs = np.stack([a, c, b]); times = np.array([0.0, 10.0, 20.0])
    kerala, bihar = frozenset({"Q1186"}), frozenset({"Q1165"})
    # same text-likeness, different states: c (0.75 < 0.80 override) starts its own event
    labels, _, _ = assign_online(vecs, times, ["o1", "o2", "o3"], 0.65, places=[kerala, bihar, bihar])
    assert labels[0] != labels[1]
    # no places known: c joins a as before
    labels2, _, _ = assign_online(vecs, times, ["o1", "o2", "o3"], 0.65)
    assert labels2[0] == labels2[1]
    # near-duplicate (0.95 >= 0.80) joins despite a state mismatch
    labels3, _, _ = assign_online(np.stack([a, b]), np.array([0.0, 5.0]), ["o1", "o2"], 0.65, places=[kerala, bihar])
    assert labels3[0] == labels3[1]
