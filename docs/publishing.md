# Publishing warp-optix to PyPI

Releases use GitHub Actions Trusted Publishing. Maintainers do not share a
PyPI token. This workflow currently releases Windows x64 / CPython 3.12 wheels.
Package metadata therefore requires Python 3.12 (`>=3.12,<3.13`) and advertises
Windows support only. Linux wheels need a separate manylinux packaging change;
broaden the metadata when those wheels and additional Python versions ship.
No source distribution is published by this workflow. PyOptiX remains a
separate dependency; the bundled CI PyOptiX wheel is not published by this project.

## One-time GitHub setup

You need administrator access to NVIDIA/warp-optix.

1. Open https://github.com/NVIDIA/warp-optix/settings/environments.
2. Click **New environment**, enter `pypi`, and click **Configure environment**.
3. Enable **Required reviewers**, add the maintainers allowed to approve releases,
   and save the protection rules. Only one listed reviewer needs to approve.
4. Leave **Prevent self-review** unchecked if the person starting a release should
   also be able to approve it. Enable it if a second person must approve.
5. Under **Deployment branches and tags**, choose **Selected branches and tags**.
   Add a **Tag** rule with the pattern `v*`.
6. No environment secret or PyPI API token is needed.

## One-time PyPI setup

You need Owner access to the existing warp-optix PyPI project.

1. Open https://pypi.org/manage/project/warp-optix/settings/publishing/.
2. Under **Add a new publisher**, select **GitHub** and enter:

   | Field | Value |
   | --- | --- |
   | PyPI project | `warp-optix` (already selected) |
   | Owner | `NVIDIA` |
   | Repository name | `warp-optix` |
   | Workflow name / filename | `release.yml` |
   | Environment name | `pypi` |

   The workflow value is the filename, not `Release` or the full file path.
3. Click **Add** and check the publisher appears on the page.
4. Under **Collaborators**, invite at least one additional trusted PyPI account
   as an Owner for project administration and recovery. That person must accept.

References:
- https://docs.pypi.org/trusted-publishers/adding-a-publisher/
- https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments

## Install the workflow

Merge `.github/workflows/release.yml` and this guide into `main` through the
repository's normal review process. Neither merging the workflow nor configuring
the publisher uploads a package. Publishing starts when a new `v*` tag is pushed,
and still requires approval in the `pypi` environment.

Release builds disable the uv, PyOptiX, and Warp kernel caches. The GitHub-hosted
build records the warp-optix wheel's SHA-256 as a workflow output; validation
checks the downloaded wheel against that digest before inspecting or staging it.
Validation also requires the tagged commit to be an ancestor of `origin/main`.

## Publish the next version

Version 0.2.0 was published manually. Do not rebuild or replace its existing wheel.
Use an unused version for the next release, for example 0.2.1 or 0.2.1a1.

1. In a release preparation branch, update `[project].version` in `pyproject.toml`
   and run `uv lock`. Review and merge those changes into `main`.
2. Confirm the version is unused on PyPI, then update your local `main`:

   ```powershell
   Set-Location C:\git3\warp-optix
   git switch main
   git pull --ff-only origin main
   git status --short
   ```

3. Confirm the checkout is clean and its version is the one you intend to publish.
   For version 0.2.1, create and push the tag:

   ```powershell
   git tag v0.2.1
   git push origin v0.2.1
   ```

   Substitute the actual version in both commands. Pushing the tag starts release
   automation; it is not a test run.
4. Open https://github.com/NVIDIA/warp-optix/actions/workflows/release.yml.
   Open the new run. Wait for the Windows build, GPU tests, and metadata validation.
5. When **Publish to PyPI** waits for approval, click **Review deployments**,
   select `pypi`, and click **Approve and deploy**. This authorizes the public upload.
6. Confirm the new release appears at https://pypi.org/project/warp-optix/.
7. After the first successful Trusted Publishing release, revoke the temporary
   account-wide token used for the initial manual upload.

## Troubleshooting

- No **Settings** or environment controls: ask a repository administrator to do
  the GitHub setup. PyPI project ownership does not grant GitHub administration.
- `invalid-publisher`: compare the four PyPI fields above with the workflow and
  repository, including the environment `pypi`.
- Tag mismatch: the tag suffix must exactly match the wheel's version. Do not
  rename or move an existing release tag to work around a mismatch.
- Commit outside `main`: merge the release changes through normal review before
  tagging. The workflow rejects tags whose commits are not in `origin/main`.
- Wheel hash mismatch: the downloaded artifact differs from the hosted build's
  output. Investigate the build, artifact, and GPU runner before starting a new release.
- File already exists: the version/file was already uploaded. Published wheels
  cannot be replaced; prepare a new version for changed package contents.
- Waiting for a GPU runner: the Windows suite uses the existing NVIDIA runner
  pool; publishing waits for its tests to complete.
