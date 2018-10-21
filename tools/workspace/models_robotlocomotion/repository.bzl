# -*- python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def models_robotlocomotion_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "pangtao22/models",
        commit = "fd6e834ffc47a362a2679c2e6fb2d05917520e99",
        sha256 = "a3a06f4978df8cd0b64683abed3f4b9f8289555a73428dc82e880af5f3ceddb3",  # noqa
        build_file = "@drake//tools/workspace/models_robotlocomotion:package.BUILD.bazel",  # noqa
        # local_repository_override = "/home/russt/models",
        mirrors = mirrors,
    )
