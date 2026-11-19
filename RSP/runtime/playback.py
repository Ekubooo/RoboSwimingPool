"""Compatibility entry point for callers that already own a MimicKit session."""


def play(env, agent):
    from RSP.app.playback import play_mimickit

    play_mimickit(env, agent)
