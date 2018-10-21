# -*- python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def models_robotlocomotion_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "pangtao22/models",
        commit = "a1312f673ed7ec823f4747675686fc2102059666",
        sha256 = "eecae797c321be1e10c52878dcbbc2a062cde87de0bd725ec25b14f857890afb",  # noqa
        build_file = "@drake//tools/workspace/models_robotlocomotion:package.BUILD.bazel",  # noqa
        mirrors = mirrors,
    )
