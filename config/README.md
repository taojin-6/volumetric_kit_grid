# Running an example

Each executable reads the versioned JSON format owned by
`volumetric_kit::core_config`: run `program --config path.json` or
`program --help`. Core validates the document before the executable opens a
GPU or output file. Paths are relative to the configuration file;
environment variables and home-directory shortcuts are not expanded. The
format's settings and defaults are documented with recon's examples, which
read it too ([recon's config/README.md](https://github.com/taojin-6/volumetric_kit_recon/blob/main/config/README.md)).

| Executable | Starting configuration | Work |
| --- | --- | --- |
| `codec_mesh` | [codec_mesh.json](examples/codec_mesh.json) | One mesh, placed in metres, and codec study |
| `codec_mesh` | [codec_mesh_sequence.json](examples/codec_mesh_sequence.json) | Numbered mesh sequence under one placement, coded frame by frame |

`codec_mesh` counts exactly: `run.frames` meshes, numbered from `run.first`,
one by default. In a mesh's `source.path`, write a literal `%` as `%%`;
literal `%` characters in the configuration file's directory are escaped
automatically when resolving a relative mesh path.
`codec_mesh_sequence.json` codes at `codec.intra_period` 30;
`codec_mesh.json`, one frame, keeps 1.

For mesh inspection, set only `codec.inspect_only: true` in the `codec` section
and omit `fusion`, `source.sdf_mode`, `source.shell_voxels` and
`source.allow_holes`; these settings are unused until mesh conversion runs.
`run` and either scale are accepted, so a range is inspected frame by frame.
An optional `output.prefix` still exports the last frame's normalized input
mesh.
