# Changelog

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
