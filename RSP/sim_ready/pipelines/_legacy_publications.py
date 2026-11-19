"""Exact recipe aliases for the reference packages released before relocation."""

# Verified against the original publication code at b98e26d, and 3832639 for G1.
# The new keys include the source identity, input and producer hashes, h and
# dependencies. Changing any of those still requires a new asset version.
LEGACY_RECIPE_DIGESTS = {
    # articulations/g1/1
    "ffa00c80a2dfbb8467528a020fa9ec36bd3b629d9c96d047f9429070541da498":
        "17abe43fbfcb109f0f2e1e0bb06cc9edf0f135076b4c1e765ed4d31f1ba629a5",
    # articulations/humanoid/2
    "7ed1d25189022aa3e01c48e9cce9377bb33ace2216246b8b08e9abbd945d87a7":
        "bb24e243a357d1af9b887fd39ff81c21442df072720efeab3b1f3e730fdf82e7",
    # containers/pbf_pool/2
    "42920f92d285f7896cecd49bb555d85bc8f9c05c76a3b0703a90aa823f531bee":
        "4705910081ceadbc1a33a7ab5f077ed9c38fe90424c8cfb340e4b3f769836b51",
    # containers/pbf_pool_wide/1
    "ec9b756a7ce152919e7505ec4dac4c027f8189e7910380d4de73bb4bc4011da2":
        "266056dec9c1205550364d1cb8fab12e2b31ced4fe66cffea23d7b6248b1844a",
    # props/heavy_box/2
    "0be11466496ea33af28e785ff7786ffac6572d2717ff39078ca82af1c9e53d93":
        "860e2b750c338098afa74c96088fbbc1bed9fa1cb960da2f25e5ef6fcdc6574c",
    # props/light_box/2
    "e604d112445acd09f2eb2cbd2b1990864b7875e6c3f4b5af7dd210da739710c1":
        "90f95f0f116e5c074b0153365e099ced59456ed36ec18b437ea1058f6a477f1e",
    # props/scene_sphere/2
    "911edc4b95980c585c910e06e9b0b9a92f98977ffee109c8980b23ad1b5a0601":
        "974f53533c411ac19ffc0594840910d529c06ffc309a996dacf50ad34174c00d",
}
