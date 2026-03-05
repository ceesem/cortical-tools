# Changelog

## 0.1.1

### Fixed

* Fixed docs GitHub Action to pin `mkdocs<2.0` and update `mkdocstrings-python` 2.x configuration (`import` → `inventories`, removed deprecated `allow_section_blank_line` option).
* Fixed Python 3.10 compatibility: `Self` from `typing` is only available in 3.11+, now imported from `typing_extensions` on 3.10.

## 0.1.0

### Added

* Lifted a series of commonly-used caveclient methods directly onto the dataset client, making them accessible without going through `client.cave`. This includes methods for chunkedgraph operations (`get_roots`, `is_latest_roots`, `get_latest_roots`, `suggest_latest_roots`), skeleton retrieval (`get_skeleton`, `skeletons_exist`), L2 cache (`get_l2_ids`, `get_l2data`), info service (`get_datastack_info`, `image_source`, `segmentation_source`, `viewer_resolution`, `image_cloudvolume`, `segmentation_cloudvolume`), JSON state (`get_state_json`, `upload_state_json`), annotation staging (`stage_annotations`, `upload_staged_annotations`), and materialization utilities (`most_recent_materialization_version`, `version_timestamp`, `latest_valid_timestamp`).

## 0.0.7

### Added

* Added `root_ids` argument to `neuroglancer_url` method to allow specifying root IDs to include in the Neuroglancer view.

### Fixed

* Fixed __repr__ methods for MicronsProdClient to return strings correctly.
* Updated `nglui` dependency version to fix string issues.

## 0.0.6

### Fixed

* Fixed a bug in `root_id_to_cell_id` that could cause incorrect results in certain conditions when using alternative lookup tables.
* Removed a non-available alternative lookup table from the v1dd_public dataset client.

## 0.0.5

### Added

* Added `cortical_tools.load_client` method that takes a dataset name (currently one of "v1dd", "v1dd_public", "microns_prod", "microns_public") and returns the corresponding dataset client. This is intended for scripts and paramterized notebooks.

## 0.0.4

### Added
 
* Added `query_synapses` method to query synapses inclusively with reference tables.
* Added optional bounds argument to get_l2_ids to limit search area.
* Added dataset-active tests for all datastacks.

### Fixed

* Fixed bug in streamline transformations for skeletons and synapses.
* Fixed bug in get_l2_ids that did not work.
* Allowed `cell_id_to_root_id` and `root_id_to_cell_id` to work with a single numeric ID.
* Suppress caveclient warnings.

## 0.0.3

### Fixed

* Improved mesh vertex lookup memory usage and performance. Should no longer crash on large meshes due to out of memory issues.

## 0.0.2

### Changed

Added additional docstrings.

## 0.0.1

First release!
