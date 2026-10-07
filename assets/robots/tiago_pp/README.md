# TIAGo++ asset provenance

The canonical v1 robot is the `pal_tiago_dual` model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_tiago_dual).

The model is not vendored into this repository. Install the pinned
`mujoco-menagerie` package and POSTMAN will resolve/download the asset into the
user cache automatically. You can override this with
`POSTMAN_TIAGO_PP_MJCF`. Preserve the model's Apache-2.0 license and source
revision when adding local assets.
