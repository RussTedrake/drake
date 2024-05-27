load("//tools/workspace:github.bzl", "github_archive")

def common_robotics_utilities_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "RussTedrake/common_robotics_utilities",
        upgrade_advice = """
        When updating, ensure that any new unit tests are reflected in
        package.BUILD.bazel and BUILD.bazel in drake. Tests may have been
        updated in RussTedrake/common_robotics_utilities/test/ or
        RussTedrake/common_robotics_utilities/CMakeLists.txt.ros2
        """,
        commit = "fedcdabc4ef3e67dbe2dafe7c5794bcfbf179334",
        sha256 = "9f89c8116a9e74cc700b811efe3d440be28b34bafb421864eef4dfb35c1dbd97",  # noqa
        build_file = ":package.BUILD.bazel",
        patches = [
            ":patches/vendor.patch",
        ],
        mirrors = mirrors,
    )
