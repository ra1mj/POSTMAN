# Scene research and selection

The first checked-in scene family is `furniture_sim`, pinned in
`third_party/scenes/furniture_sim/SOURCE.md`. Its simple table, hinged cabinet,
sliding cabinet, oven, and microwave are useful for increasing task difficulty
without changing the TIAGo++ model.

The scene composer uses MuJoCo `MjSpec.attach` rather than nested XML includes.
That matters for Menagerie assets because robot meshes use a model-local
`assets/` directory. The resulting composed model is compiled in memory and
can be fed directly to the MuJoCo adapter.

The mjlab manipulation repository is kept as a reference for Reach/Lift and
RGB object-task design. Its robot set is arm-centric, so it is not copied into
the POSTMAN core. EBiM's MuJoCo tasks are a later benchmark integration because
they bring a larger teleoperation/evaluation dependency surface.
