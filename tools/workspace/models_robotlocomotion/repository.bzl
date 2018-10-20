# -*- python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def models_robotlocomotion_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "RobotLocomotion/models",
        commit = "10e63006c49f0ad3d98f387856b42163c1995be6",
        sha256 = "f44cbb7e64759b77c0665c48f482c627e02b78c23bc0709a5fc305aecf880b99",  # noqa
        build_file = "@drake//tools/workspace/models_robotlocomotion:package.BUILD.bazel",  # noqa
        mirrors = mirrors,
    )
